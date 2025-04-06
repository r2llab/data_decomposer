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
from implementations.symphony.utils.cost_tracker import CostTracker

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, index_path: Optional[Path] = None, openai_api_key: Optional[str] = None, 
                 cost_tracker: Optional[CostTracker] = None):
        """
        Initialize the pipeline components.
        
        Args:
            index_path: Optional path to load an existing discovery index
            openai_api_key: Optional OpenAI API key for decomposition and aggregation
            cost_tracker: Optional cost tracker instance to track API usage
        """
        print(index_path)
        self.cost_tracker = cost_tracker or CostTracker()
        self.embedder = AutoEmbedder(api_key=openai_api_key)
        self.discovery = Discovery(embedder=self.embedder, index_path=index_path)
        self.executor = Executor(api_key=openai_api_key, cost_tracker=self.cost_tracker)
        self.decomposer = Decomposer(api_key=openai_api_key, cost_tracker=self.cost_tracker)
        self.aggregator = Aggregator(api_key=openai_api_key, cost_tracker=self.cost_tracker)
        
        # Add a set to track document sources used, similar to ReSP
        self.document_sources = set()

    def run_query(self, query: str) -> Dict[str, Any]:
        """
        Run the complete pipeline on a given query.
        
        Args:
            query: The natural language query string
            
        Returns:
            Dict containing the answer and metadata
        """

        logger.info(f"\n\n=== Processing Query: {query} ===")
        
        # Reset document sources for new query
        self.document_sources = set()
        
        # 1. Discovery phase
        logger.info("\n--- Discovery Phase ---")
        relevant_items = self._discover(query)
        logger.info(f"Found {len(relevant_items)} relevant items:")
        for idx, item in enumerate(relevant_items, 1):
            logger.info(f"Item {idx}: {item.get('type')} - Score: {item.get('relevance_score', 'N/A')}")
            # Handle both data and content fields
            content = item.get('data') if item.get('data') is not None else item.get('content')
            
            # Track document sources
            if 'metadata' in item and 'source' in item['metadata']:
                source = item['metadata']['source']
                if source and source != "unknown":
                    self.document_sources.add(source)
            elif 'source' in item:
                source = item['source']
                if source and source != "unknown":
                    self.document_sources.add(source)
            
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
                "source": None,
                "document_sources": []
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
            logger.info(f"Executing query: {query}")
            logger.info(f"On item: {relevant_items[0].get('data', relevant_items[0].get('content', 'No content'))[:100]}...")
            result = self._execute(query, relevant_items[0])
            
            # Add document sources to the result
            result['document_sources'] = list(self.document_sources)
            logger.info(f"Document sources used: {self.document_sources}")
            
            return result
            
        # 3. Execute each sub-query
        logger.info("\n--- Execution Phase ---")
        sub_results = []
        for sub_query in decomposition["sub_queries"]:
            target_idx = sub_query["target_item_index"] - 1
            if target_idx < len(relevant_items):
                logger.info(f"Executing sub-query '{sub_query['sub_query']}' on item {relevant_items[target_idx]['data']}")
                
                # Track document source from the target item
                if 'metadata' in relevant_items[target_idx] and 'source' in relevant_items[target_idx]['metadata']:
                    source = relevant_items[target_idx]['metadata']['source']
                    if source and source != "unknown":
                        self.document_sources.add(source)
                elif 'source' in relevant_items[target_idx]:
                    source = relevant_items[target_idx]['source']
                    if source and source != "unknown":
                        self.document_sources.add(source)
                
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
                "source": None,
                "document_sources": list(self.document_sources)
            }
            
        # 4. Aggregate results
        if len(sub_results) == 1:
            logger.info("\n--- Single Result - No Aggregation Needed ---")
            sub_results[0]['document_sources'] = list(self.document_sources)
            return sub_results[0]
            
        logger.info("\n--- Aggregation Phase ---")
        final_result = self.aggregator.aggregate_results(
            original_query=query,
            sub_queries=decomposition["sub_queries"],
            sub_results=sub_results,
            aggregation_strategy=decomposition["aggregation_strategy"]
        )
        logger.info(f"Final Answer: {final_result['answer']}")
        
        # Add document sources to the final result
        final_result['document_sources'] = list(self.document_sources)
        logger.info(f"Document sources used: {self.document_sources}")
        
        return final_result

    def _discover(self, query: str) -> List[Dict[str, Any]]:
        """Use the discovery module to find relevant items."""
        return self.discovery.discover(query, k=4, min_score=0.3)

    def _execute(self, query: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the query on a single item."""
        # Track document source from the item if available
        if 'metadata' in item and 'source' in item['metadata']:
            source = item['metadata']['source']
            if source and source != "unknown":
                self.document_sources.add(source)
        elif 'source' in item:
            source = item['source']
            if source and source != "unknown":
                self.document_sources.add(source)
                
        return self.executor.execute_query(query, item)
    
    def embed_data(self, data_dir: str, batch_size: int = 16, output_dir: Optional[str] = None) -> np.ndarray:
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
        dataset = CrossModalDataset.from_directory(data_dir, batch_size=batch_size)
        logger.info(f"Loaded dataset with {len(dataset)} items")
        
        # Process embeddings in batches
        logger.info("Starting embedding extraction")
        all_embeddings = []
        
        # Process in batches using the dataset's iterator
        items_to_process = []
        processed_count = 0
        total_items = len(dataset)
        
        for item in dataset:
            # Truncate text to max 8000 tokens (approximate limit for safety)
            if len(item.split()) > 8000:
                item = ' '.join(item.split()[:8000])
                
            items_to_process.append(item)
            
            # When we have a full batch or at the end, process it
            if len(items_to_process) >= batch_size or processed_count + len(items_to_process) == total_items:
                try:
                    # Get embeddings for the batch
                    embeddings = self.embedder.embed_text(items_to_process)
                    all_embeddings.append(embeddings)
                    
                    processed_count += len(items_to_process)
                    logger.info(f"Processed items {processed_count}/{total_items} ({(processed_count/total_items)*100:.1f}%)")
                    
                    # Clear the batch
                    items_to_process = []
                    
                    # Add a small delay between batches to respect rate limits
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    # Process items one by one as fallback
                    for single_item in items_to_process:
                        try:
                            embedding = self.embedder.embed_text([single_item])
                            all_embeddings.append(embedding)
                            processed_count += 1
                            logger.info(f"Processed item {processed_count}/{total_items} (fallback mode)")
                            time.sleep(1)  # Delay between fallback items
                        except Exception as inner_e:
                            logger.error(f"Error processing item in fallback mode: {str(inner_e)}")
                            # Use zero vector as fallback
                            if all_embeddings:
                                fallback = np.zeros((1, all_embeddings[0].shape[1]))
                            else:
                                fallback = np.zeros((1, 1536))  # Default embedding size
                            all_embeddings.append(fallback)
                            processed_count += 1
                    
                    # Clear the batch
                    items_to_process = []
        
        # Concatenate all embeddings
        if all_embeddings:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            logger.info(f"Generated embeddings with shape {all_embeddings.shape}")
        else:
            logger.error("No embeddings were generated")
            all_embeddings = np.zeros((len(dataset), 1536))
        
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