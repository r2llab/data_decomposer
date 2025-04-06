from pathlib import Path
import numpy as np
import logging
import time
from typing import Any, Dict, Optional, List
from core.base_implementation import BaseImplementation
from .core import ReSPPipeline
from .embeddings import AutoEmbedder
from .embeddings.dataset import CrossModalDataset
from .discovery.index import VectorIndex
from .utils.cost_tracker import CostTracker
import os
from difflib import SequenceMatcher
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReSPImplementation(BaseImplementation):
    """ReSP implementation of the BaseImplementation interface."""
    
    def initialize(self) -> None:
        """Initialize ReSP resources."""
        print("Initializing ReSP")
        
        # Extract config values
        self.openai_api_key = self.config.get('openai_api_key')
        self.index_path = self.config.get('index_path')
        if self.index_path:
            self.index_path = Path(self.index_path)
        
        # Create cost tracker
        self.cost_tracker = CostTracker()
        
        # Source relevance tracking
        self.ground_truth_answer = None
        self.source_relevance_scores = []
            
        # If data path is provided and no index path, process the data first
        data_path = self.config.get('data_path')
        if data_path and not self.index_path:
            print(f"Indexing data from {data_path}")
            
            # Use base data directory for storing embeddings and index
            # Get the base data directory (typically "data")
            base_data_dir = Path(data_path).parts[0]
            
            # Create paths for embeddings and index
            embeddings_dir = Path(base_data_dir) / 'embeddings'
            index_dir = Path(base_data_dir) / 'index'
            
            # Ensure directories exist
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            index_dir.mkdir(parents=True, exist_ok=True)
            
            embeddings_path = embeddings_dir / 'embeddings.npy'
            
            # Check if embeddings file exists AND is not empty
            embeddings_need_generation = True
            if embeddings_path.exists():
                # Check if embeddings file is empty
                try:
                    embeddings = np.load(str(embeddings_path))
                    if embeddings.shape[0] > 0:
                        print(f"Using existing embeddings file: {embeddings_path} with {embeddings.shape[0]} embeddings")
                        embeddings_need_generation = False
                    else:
                        print(f"Embeddings file exists but is empty. Regenerating embeddings.")
                        # Optionally delete the empty file
                        embeddings_path.unlink()
                except Exception as e:
                    print(f"Error reading embeddings file: {e}. Regenerating embeddings.")
                    # Optionally delete the corrupted file
                    embeddings_path.unlink()
            
            # Generate embeddings if needed
            if embeddings_need_generation:
                print(f"Generating embeddings for {data_path}")
                embedder = AutoEmbedder(api_key=self.openai_api_key)
                embeddings = self.embed_data(
                    data_dir=data_path,
                    embedder=embedder,
                    batch_size=self.config.get('batch_size', 16),
                    output_dir=str(embeddings_dir)
                )
            
            # Create index
            print(f"Indexing data from {data_path} to {index_dir}")
            self.index_data(
                data_dir=data_path,
                embeddings_path=str(embeddings_path),
                output_dir=str(index_dir)
            )
            self.index_path = index_dir
            
        # Initialize the pipeline with configuration
        self.pipeline = ReSPPipeline(
            index_path=self.index_path,
            openai_api_key=self.openai_api_key,
            max_iterations=self.config.get('max_iterations', 5),
            log_file="logs/resp_pipeline.log",
            cost_tracker=self.cost_tracker
        )
        
        # Patch the pipeline to track source relevance
        self._patch_pipeline_for_source_relevance()
    
    def _patch_pipeline_for_source_relevance(self):
        """Patch pipeline methods to track source relevance."""
        original_retrieve = self.pipeline._retrieve
        
        def wrapped_retrieve(query):
            """Wrapper for retrieve that tracks source relevance."""
            retrieved_docs = original_retrieve(query)
            
            # Calculate relevance scores if ground truth is available
            if self.ground_truth_answer and retrieved_docs:
                for doc in retrieved_docs:
                    content = doc.get('content')
                    if content is not None:
                        relevance = self._calculate_text_similarity(content, self.ground_truth_answer)
                        self.source_relevance_scores.append(relevance)
                        print(f"Source relevance score: {relevance:.4f}")
            
            return retrieved_docs
        
        # Replace the method
        self.pipeline._retrieve = wrapped_retrieve
    
    def _calculate_text_similarity(self, content: Any, reference: str) -> float:
        """
        Calculate the similarity between content and reference text.
        Handles different content types including DataFrames.
        
        Args:
            content: The content to compare (could be text, DataFrame, etc.)
            reference: The reference text to compare against
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Handle pandas DataFrame
        if isinstance(content, pd.DataFrame):
            try:
                # Convert DataFrame to string representation
                # Limit to first 10 rows for performance
                df_sample = content.head(10)
                text = df_sample.to_string()
                return SequenceMatcher(None, text, reference).ratio()
            except Exception as e:
                logger.error(f"Error processing DataFrame for similarity: {e}")
                return 0.0
        # Handle string content
        elif isinstance(content, str):
            return SequenceMatcher(None, content, reference).ratio()
        # Handle other types by converting to string
        else:
            try:
                text = str(content)
                return SequenceMatcher(None, text, reference).ratio()
            except Exception as e:
                logger.error(f"Error converting content to string for similarity: {e}")
                return 0.0
    
    def process_query(self, query: str, ground_truth_answer: Optional[str] = None) -> Any:
        """Process a query using ReSP.
        
        Args:
            query: The query string to process
            ground_truth_answer: Optional ground truth answer for relevance scoring
            
        Returns:
            Dict containing the answer and metadata
        """
        # Reset query-specific cost tracking and source relevance tracking
        self.cost_tracker.reset_query_stats()
        self.ground_truth_answer = ground_truth_answer
        self.source_relevance_scores = []
        
        # Log the start of processing
        logger.info(f"Processing query: {query}")
        
        # Run the query through the pipeline
        result = self.pipeline.run_query(query)
        
        # Ensure document sources are included in the output and in the correct format
        if 'document_sources' not in result:
            result['document_sources'] = []
            
        # Convert document_sources to a properly formatted list
        # This ensures sources are properly formatted even if they were gathered elsewhere
        document_sources = result.get('document_sources', [])
        if isinstance(document_sources, set):
            document_sources = list(document_sources)
        
        # Remove any None or empty values and normalize paths
        document_sources = [src for src in document_sources if src]
        
        # Log the extracted answer and sources
        logger.info(f"Answer: {result.get('answer', '')}")
        logger.info(f"Sources: {document_sources}")
            
        # Add a structured sources field for improved clarity in output
        result['source_documents'] = []
        for source in document_sources:
            # Skip empty or None sources
            if not source:
                continue
                
            # Convert source to string if it's not already
            source = str(source)
            
            # Determine the source type
            source_type = 'table' if '.csv' in source.lower() else 'text'
            
            # Add to source_documents list
            result['source_documents'].append({
                'id': source,
                'type': source_type
            })
        
        # Add formatted document_sources back to result
        result['document_sources'] = document_sources
        
        # Add source relevance score if ground truth was provided
        if self.ground_truth_answer and self.source_relevance_scores:
            # Calculate average relevance score
            avg_relevance = sum(self.source_relevance_scores) / len(self.source_relevance_scores)
            # Calculate max relevance score
            max_relevance = max(self.source_relevance_scores) if self.source_relevance_scores else 0.0
            
            result['source_relevance_score'] = {
                'average': float(avg_relevance),
                'maximum': float(max_relevance),
                'scores': [float(score) for score in self.source_relevance_scores]
            }
            print(f"Source relevance: avg={avg_relevance:.4f}, max={max_relevance:.4f}")
        
        # Log cost summary
        cost_summary = self.cost_tracker.get_query_summary()
        logger.info(f"Query cost: ${cost_summary['query_cost']:.6f}")
        logger.info(f"Total tokens: {cost_summary['query_tokens']}")
        logger.info(f"API calls: {cost_summary['query_calls']}")
        
        return result
    
    def embed_data(self, data_dir: str, embedder: Optional[AutoEmbedder] = None, batch_size: int = 16, output_dir: Optional[str] = None) -> np.ndarray:
        """
        Load data from directory and embed all items.
        
        Args:
            data_dir: Directory containing the dataset files
            embedder: Optional embedder to use (if None, creates a new one)
            batch_size: Size of batches for processing
            output_dir: Optional directory to save embeddings. If provided, saves as 'embeddings.npy'
            
        Returns:
            np.ndarray: Array of embeddings for all items
        """
        # Use provided embedder or create a new one
        if embedder is None:
            embedder = AutoEmbedder(api_key=self.openai_api_key)
            
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
            # # Truncate text to max 8000 tokens (approximate limit for safety)
            # if len(item.split()) > 8000:
            #     item = ' '.join(item.split()[:8000])
                
            items_to_process.append(item)
            
            # When we have a full batch or at the end, process it
            if len(items_to_process) >= batch_size or processed_count + len(items_to_process) == total_items:
                try:
                    # Get embeddings for the batch
                    embeddings = embedder.embed_text(items_to_process)
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
                            embedding = embedder.embed_text([single_item])
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
        Index data from a directory, saving embeddings and creating a vector index.
        
        Args:
            data_dir: Path to data directory
            embeddings_path: Path to save/load embeddings
            output_dir: Output directory for index files
            
        Returns:
            VectorIndex: Vector index for search
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if embeddings file exists and is not empty
        embeddings_exists = os.path.exists(embeddings_path) and os.path.getsize(embeddings_path) > 0
        
        if embeddings_exists:
            logger.info(f"Loading existing embeddings from {embeddings_path}")
            try:
                # Load existing embeddings
                embeddings = np.load(embeddings_path)
                # Get the dataset for metadata
                logger.info(f"Loading dataset from {data_dir}")
                dataset = CrossModalDataset.from_directory(data_dir, batch_size=64)  # Increased batch size
                
                # Check if dataset and embeddings match in size
                if len(dataset) != len(embeddings):
                    logger.warning(f"Dataset size ({len(dataset)}) does not match embeddings size ({len(embeddings)})")
                    logger.warning("Using existing embeddings but adjusting sizes to match")
                    
                    # If dataset is larger, we need more embeddings
                    if len(dataset) > len(embeddings):
                        logger.warning(f"Dataset is larger than embeddings file. Using only the first {len(embeddings)} items.")
                        # Truncate dataset to match embeddings
                        dataset.items = dataset.items[:len(embeddings)]
                    # If embeddings are more, truncate them
                    else:
                        logger.warning(f"Embeddings file is larger than dataset. Using only the first {len(dataset)} embeddings.")
                        # Truncate embeddings to match dataset
                        embeddings = embeddings[:len(dataset)]
            except Exception as e:
                logger.error(f"Error loading embeddings: {e}")
                logger.info("Regenerating embeddings")
                
                # # Load dataset and generate embeddings
                # logger.info(f"Loading dataset from {data_dir}")
                # dataset = CrossModalDataset.from_directory(data_dir, batch_size=128)  # Larger batch size for efficiency
                
                # logger.info(f"Generating embeddings for {len(dataset)} items")
                # # Create embedder with larger batch size and token limits
                # embedder = AutoEmbedder(api_key=self.openai_api_key, max_tokens=8192)
                # try:
                #     # Use the optimized embed_dataset method with larger batch size
                #     logger.info("Using optimized batch embedding process")
                #     embeddings = embedder.embed_dataset(dataset, batch_size=128)
                    
                #     # Save embeddings
                #     np.save(embeddings_path, embeddings)
                #     logger.info(f"Saved embeddings to {embeddings_path}")
                # except Exception as embed_err:
                #     logger.error(f"Error during embedding: {embed_err}")
                #     # Fall back to the embed_data method which is more resilient
                #     logger.info("Falling back to individual item embedding")
                #     embeddings = self.embed_data(
                #         data_dir=data_dir,
                #         embedder=embedder,
                #         batch_size=64,  # Increased batch size
                #         output_dir=os.path.dirname(embeddings_path)
                #     )
        # else:
        #     # Embeddings don't exist, generate them
        #     logger.info(f"Embeddings file not found at {embeddings_path}. Generating new embeddings.")
            
        #     # Load dataset and generate embeddings
        #     logger.info(f"Loading dataset from {data_dir}")
        #     dataset = CrossModalDataset.from_directory(data_dir, batch_size=128)  # Increased batch size
            
        #     logger.info(f"Starting embedding extraction with {len(dataset)} items")
        #     # Create embedder with larger batch size and token limits
        #     embedder = AutoEmbedder(api_key=self.openai_api_key, max_tokens=8192)
        #     try:
        #         # Try using the new optimized method with larger batch size
        #         logger.info("Using optimized batch embedding process")
        #         embeddings = embedder.embed_dataset(dataset, batch_size=128)
                
        #         # Save embeddings
        #         np.save(embeddings_path, embeddings)
        #         logger.info(f"Saved embeddings to {embeddings_path} ({len(embeddings)} embeddings)")
        #     except Exception as e:
        #         logger.error(f"Error using embed_dataset, falling back to embed_data: {e}")
        #         # Fall back to the old method which processes one-by-one if needed
        #         logger.info("Falling back to individual item embedding")
        #         embeddings = self.embed_data(
        #             data_dir=data_dir,
        #             embedder=embedder,
        #             batch_size=64,  # Increased batch size
        #             output_dir=os.path.dirname(embeddings_path)
        #         )
        
        # Create index using embeddings
        logger.info(f"Creating index with {len(embeddings)} embeddings")
        
        # Create a new index with the right dimension
        index = VectorIndex(dimension=embeddings.shape[1])
        
        # Add items to the index
        index.add_items(
            items=dataset.items,  # Keep all fields including source, id, etc.
            embeddings=embeddings
        )
        
        # Save index
        logger.info(f"Saving index to {output_dir}")
        index.save(Path(output_dir))
        logger.info(f"Index saved successfully")
        
        return index
    
    def cleanup(self) -> None:
        """Cleanup ReSP resources."""
        # Print cost summary
        cost_summary = self.cost_tracker.get_cost_summary()
        print("\nTotal API usage summary:")
        print(f"Total cost: ${cost_summary['total_cost']:.6f}")
        print(f"Total tokens: {cost_summary['total_tokens']}")
        print(f"Total API calls: {cost_summary['total_calls']}")
        print("\nModel breakdown:")
        for model, stats in cost_summary['models'].items():
            print(f"  {model}: ${stats['cost']:.6f} ({stats['calls']} calls)")
        print("\nEndpoint breakdown:")
        for endpoint, stats in cost_summary['endpoints'].items():
            print(f"  {endpoint}: ${stats['cost']:.6f} ({stats['calls']} calls)")
        pass
