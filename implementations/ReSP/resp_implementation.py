from pathlib import Path
from typing import Any, Dict, Optional
from core.base_implementation import BaseImplementation
from .core import ReSPPipeline

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
            
        # Initialize the pipeline with configuration
        self.pipeline = ReSPPipeline(
            index_path=self.index_path,
            openai_api_key=self.openai_api_key,
            max_iterations=self.config.get('max_iterations', 5)
        )
    
    def process_query(self, query: str) -> Any:
        """Process a query using ReSP.
        
        Args:
            query: The query string to process
            
        Returns:
            Dict containing the answer and metadata
        """
        return self.pipeline.run_query(query)
    
    def cleanup(self) -> None:
        """Cleanup ReSP resources."""
        # Currently no cleanup needed for ReSP
        pass
