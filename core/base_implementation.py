from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseImplementation(ABC):
    """Base class that all implementations must inherit from."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the implementation with configuration.
        
        Args:
            config: Configuration dictionary containing implementation-specific settings
        """
        self.config = config
        self.initialize()
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize any resources needed by the implementation."""
        pass
    
    @abstractmethod
    def process_query(self, query: str, ground_truth_answer: Optional[str] = None) -> Any:
        """Process a query using the implementation.
        
        Args:
            query: The query string to process
            ground_truth_answer: Optional ground truth answer for relevance scoring
            
        Returns:
            Implementation-specific response
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup any resources used by the implementation."""
        pass 