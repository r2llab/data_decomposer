from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import time

# Import Symphony components
from implementations.symphony.embeddings import AutoEmbedder
from implementations.symphony.embeddings.dataset import CrossModalDataset
from implementations.symphony.discovery import Discovery
from implementations.symphony.discovery.index import VectorIndex
from implementations.symphony.execution import Executor
from implementations.symphony.execution.aggregator import Aggregator
from implementations.symphony.decomposition import Decomposer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, index_path: Optional[Path] = None, openai_api_key: Optional[str] = None):
        """
        Initialize the pipeline components.
        
        Args:
            index_path: Optional path to load an existing discovery index
            openai_api_key: Optional OpenAI API key for decomposition and aggregation
        """
        print(index_path)
        self.embedder = AutoEmbedder(api_key=openai_api_key)
        self.discovery = Discovery(embedder=self.embedder, index_path=index_path)
        self.executor = Executor(api_key=openai_api_key)
        self.decomposer = Decomposer(api_key=openai_api_key)
        self.aggregator = Aggregator(api_key=openai_api_key)

    def run_query(self, query: str) -> Dict[str, Any]:
        """
        Run the complete pipeline on a given query.
        
        Args:
            query: The natural language query string
            
        Returns:
            Dict containing the answer and metadata
        """

        logger.info(f"\n\n=== Processing Query: {query} ===")
        
        # 1. Discovery phase
        logger.info("\n--- Discovery Phase ---")
        relevant_items = self._discover(query)
        logger.info(f"Found {len(relevant_items)} relevant items:")
        for idx, item in enumerate(relevant_items, 1):
            logger.info(f"Item {idx}: {item.get('type')} - Score: {item.get('relevance_score', 'N/A')}")
            # Handle both data and content fields
            content = item.get('data') if item.get('data') is not None else item.get('content')
            if content is not None:  # Check if content exists
                if isinstance(content, str):
                    logger.info(f"Content: {content[:100]}...")
                elif isinstance(content, pd.DataFrame):
                    # For DataFrames, show first few rows and columns
                    preview = str(content.head(2).to_string(max_cols=3))
                    logger.info(f"Content (DataFrame preview):\n{preview}")
                else:
                    # For other types, convert to string safely
                    preview = str(content)[:100]
                    logger.info(f"Content: {preview}...")
        
        if not relevant_items:
            logger.warning("No relevant items found")
            return {
                "answer": "I could not find any relevant information to answer your question.",
                "confidence": 0.0,
                "source_type": None,
                "source": None
            }
        
        # 2. Decomposition phase - break down complex queries
        logger.info("\n--- Decomposition Phase ---")
        decomposition = self.decomposer.decompose_query(query, relevant_items)
        logger.info(f"Requires decomposition: {decomposition['requires_decomposition']}")
        if decomposition["requires_decomposition"]:
            logger.info("Sub-queries:")
            for sq in decomposition["sub_queries"]:
                logger.info(f"- {sq['sub_query']}")
        
        if not decomposition["requires_decomposition"]:
            # Simple query - just execute on the most relevant item
            logger.info("Simple query - executing on most relevant item")
            return self._execute(query, relevant_items[0])
            
        # 3. Execute each sub-query
        logger.info("\n--- Execution Phase ---")
        sub_results = []
        for sub_query in decomposition["sub_queries"]:
            target_idx = sub_query["target_item_index"] - 1
            if target_idx < len(relevant_items):
                logger.info(f"Executing sub-query '{sub_query['sub_query']}' on item {relevant_items[target_idx]['data']}")
                result = self._execute(
                    sub_query["sub_query"],
                    relevant_items[target_idx]
                )
                logger.info(f"Result: {result['answer']}")
                sub_results.append(result)
                
        if not sub_results:
            logger.warning("No sub-queries executed successfully")
            return {
                "answer": "Failed to execute any sub-queries successfully.",
                "confidence": 0.0,
                "source_type": None,
                "source": None
            }
            
        # 4. Aggregate results
        if len(sub_results) == 1:
            logger.info("\n--- Single Result - No Aggregation Needed ---")
            return sub_results[0]
            
        logger.info("\n--- Aggregation Phase ---")
        final_result = self.aggregator.aggregate_results(
            original_query=query,
            sub_queries=decomposition["sub_queries"],
            sub_results=sub_results,
            aggregation_strategy=decomposition["aggregation_strategy"]
        )
        logger.info(f"Final Answer: {final_result['answer']}")
        
        return final_result

    def _discover(self, query: str) -> List[Dict[str, Any]]:
        """Use the discovery module to find relevant items."""
        return self.discovery.discover(query, k=3, min_score=0.3)

    def _execute(self, query: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the query on a single item."""
        return self.executor.execute_query(query, item)
    
    def embed_data(self, data_dir: str, batch_size: int = 1, output_dir: Optional[str] = None) -> np.ndarray:
        """
        Load data from directory and embed all items.
        
        Args:
            data_dir: Directory containing the dataset files
            batch_size: Size of batches for processing
            output_dir: Optional directory to save embeddings. If provided, saves as 'embeddings.npy'
            
        Returns:
            np.ndarray: Array of embeddings for all items
        """
        
        # Load dataset
        logger.info(f"Loading dataset from {data_dir}")
        dataset = CrossModalDataset.from_directory(data_dir, batch_size=1)  # Process one at a time
        logger.info(f"Loaded dataset with {len(dataset)} items")
        
        # Process embeddings one at a time with delays
        logger.info("Starting embedding extraction")
        all_embeddings = []
        
        for i in range(len(dataset)):
            try:
                # Get single item and serialize it properly
                item = dataset[i]
                
                # Truncate text to max 8000 tokens (approximate limit for safety)
                if len(item.split()) > 8000:
                    item = ' '.join(item.split()[:8000])
                
                # Get embedding with retry logic
                max_retries = 5
                current_retry = 0
                while current_retry < max_retries:
                    try:
                        # Ensure item is a string before embedding
                        if isinstance(item, str):
                            text_to_embed = item
                        else:
                            logger.warning(f"Item {i+1} is not a string, attempting to convert...")
                            text_to_embed = str(item)
                            
                        embedding = self.embedder.embed_text([text_to_embed])
                        all_embeddings.append(embedding)
                        logger.info(f"Processed item {i+1}/{len(dataset)} successfully")
                        # Add delay between successful embeddings
                        time.sleep(3)  # 3 second delay between items
                        break
                    except Exception as e:
                        current_retry += 1
                        wait_time = min(2 ** current_retry, 60)  # Exponential backoff up to 60 seconds
                        logger.warning(f"Retry {current_retry}/{max_retries} - Waiting {wait_time} seconds... Error: {str(e)}")
                        if "400" in str(e):  # Bad request error
                            logger.error(f"Bad request error for item {i+1}. Content type: {type(item)}")
                            logger.error(f"Content preview: {str(item)[:200]}")
                        time.sleep(wait_time)
                        
                if current_retry == max_retries:
                    logger.error(f"Failed to process item {i+1} after {max_retries} retries")
                    # Use zero vector as fallback
                    if all_embeddings:
                        fallback = np.zeros_like(all_embeddings[0])
                    else:
                        fallback = np.zeros((1, 1536))  # Default embedding size for text-embedding-3-small
                    all_embeddings.append(fallback)
                    
            except Exception as e:
                logger.error(f"Error processing item {i+1}: {str(e)}")
                logger.error(f"Item type: {type(item)}")
                # Use zero vector as fallback
                if all_embeddings:
                    fallback = np.zeros_like(all_embeddings[0])
                else:
                    fallback = np.zeros((1, 1536))  # Default embedding size for text-embedding-3-small
                all_embeddings.append(fallback)
        
        # Concatenate all embeddings
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        logger.info(f"Generated embeddings with shape {all_embeddings.shape}")
        
        # Save if output directory is provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / 'embeddings.npy'
            np.save(output_path, all_embeddings)
            logger.info(f"Saved embeddings to {output_path}")
        
        return all_embeddings

    def index_data(self, data_dir: str, embeddings_path: str, output_dir: str):
        """
        Index data items and their embeddings for discovery.
        
        Args:
            data_dir: Directory containing the dataset files
            embeddings_path: Path to the embeddings.npy file
            output_dir: Directory to save the index
        """
        # Load dataset
        logger.info(f"Loading dataset from {data_dir}")
        dataset = CrossModalDataset.from_directory(data_dir)
        logger.info(f"Loaded dataset with {len(dataset)} items")
        
        # Load embeddings
        logger.info(f"Loading embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
        # Ensure embeddings are float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype('float32'))
        logger.info(f"Loaded embeddings with shape {embeddings.shape}")
        
        # Create and save index
        logger.info("Creating vector index")
        index = VectorIndex(dimension=embeddings.shape[1])
        index.add_items(
            items=[{"data": item["data"], "type": item["type"]} for item in dataset.items],
            embeddings=embeddings
        )
        
        # Save index
        logger.info(f"Saving index to {output_dir}")
        index.save(Path(output_dir))
        logger.info("Index saved successfully")