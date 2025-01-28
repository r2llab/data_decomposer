from typing import List, Dict, Any, Optional
from ..embeddings import AutoEmbedder
from ..embeddings.dataset import CrossModalDataset
from ..discovery import Discovery
from ..discovery.index import VectorIndex
from ..execution import Executor
from ..execution.aggregator import Aggregator
from ..decomposition import Decomposer
from pathlib import Path
import logging
import numpy as np
import pandas as pd

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
        # 1. Discovery phase - find relevant items
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
        return self.discovery.discover(query, k=5, min_score=0.5)

    def _execute(self, query: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the query on a single item."""
        return self.executor.execute_query(query, item)
    
    def embed_data(self, data_dir: str, batch_size: int = 32, output_dir: Optional[str] = None) -> np.ndarray:
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
        dataset = CrossModalDataset.from_directory(data_dir)
        logger.info(f"Loaded dataset with {len(dataset)} items")
        
        # Process embeddings in batches
        logger.info("Starting embedding extraction")
        all_embeddings = []
        
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            embeddings = self.embedder.embed_text(batch)
            all_embeddings.append(embeddings)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(dataset)-1)//batch_size + 1}")
        
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