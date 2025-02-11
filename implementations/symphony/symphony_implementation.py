from pathlib import Path
from typing import Any, Dict, Optional
from core.base_implementation import BaseImplementation
from .core import Pipeline

class SymphonyImplementation(BaseImplementation):
    """Symphony implementation of the BaseImplementation interface."""
    
    def initialize(self) -> None:
        print("Initializing Symphony")
        """Initialize Symphony resources."""
        # Extract config values
        self.openai_api_key = self.config.get('openai_api_key')
        self.index_path = self.config.get('index_path')
        if self.index_path:
            self.index_path = Path(self.index_path)
        
        # If data path is provided and no index path, process the data first
        data_path = self.config.get('data_path')
        if data_path and not self.index_path:
            print(f"Indexing data from {data_path}")
            embeddings_dir = Path(data_path) / 'embeddings'
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            embeddings_path = embeddings_dir / 'embeddings.npy'
            
            # Create temporary pipeline for embedding generation
            temp_pipeline = Pipeline(openai_api_key=self.openai_api_key)
            
            # Generate embeddings if they don't exist
            if not embeddings_path.exists():
                print(f"Generating embeddings for {data_path}")
                embeddings = temp_pipeline.embed_data(
                    data_dir=data_path,
                    batch_size=self.config.get('batch_size', 32),
                    output_dir=str(embeddings_dir)
                )
            
            # Create index
            index_dir = Path(data_path) / 'index'
            index_dir.mkdir(parents=True, exist_ok=True)
            print(f"Indexing data from {data_path} to {index_dir}")
            temp_pipeline.index_data(
                data_dir=data_path,
                embeddings_path=str(embeddings_path),
                output_dir=str(index_dir)
            )
            self.index_path = index_dir

        # Initialize the main pipeline with the final index path
        self.pipeline = Pipeline(
            index_path=self.index_path,
            openai_api_key=self.openai_api_key
        )
        
        
        
    
    def process_query(self, query: str) -> Any:
        """Process a query using Symphony.
        
        Args:
            query: The query string to process
            
        Returns:
            Dict containing the answer and metadata
        """
        return self.pipeline.run_query(query)
    
    def cleanup(self) -> None:
        """Cleanup Symphony resources."""
        # Currently no cleanup needed for Symphony
        pass 