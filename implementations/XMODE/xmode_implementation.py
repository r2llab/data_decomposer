from pathlib import Path
from typing import Any, Dict, Optional
from core.base_implementation import BaseImplementation
from .core import Pipeline

class XMODEImplementation(BaseImplementation):
    """XMODE implementation of the BaseImplementation interface."""
    
    def initialize(self) -> None:
        """Initialize XMODE resources."""
        print("Initializing XMODE")
        self.openai_api_key = self.config.get('openai_api_key')
        self.index_path = self.config.get('index_path')
        self.langchain_api_key = self.config.get('langchain_api_key')
        
        if not self.openai_api_key:
            raise ValueError("openai_api_key is required in config")
        if not self.index_path:
            raise ValueError("index_path is required in config")
            
        self.pipeline = Pipeline(
            index_path=Path(self.index_path),
            openai_api_key=self.openai_api_key,
            langchain_api_key=self.langchain_api_key
        )
    
    def process_query(self, query: str, ground_truth_answer: Optional[str] = None) -> Dict[str, Any]:
        """Process a query using XMODE.
        
        Args:
            query: The query string to process
            ground_truth_answer: Optional ground truth answer for relevance scoring
            
        Returns:
            Dict containing the answer and metadata
        """
        print(f"Running query: {query}")
        result = self.pipeline.run_query(query)
        
        # Ensure required fields exist
        if "answer" not in result:
            result["answer"] = "No answer found"
        
        if "document_sources" not in result:
            result["document_sources"] = []
            
        return result
    
    def cleanup(self) -> None:
        """Cleanup XMODE resources."""
        # Currently no cleanup needed for XMODE
        pass 
