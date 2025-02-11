from typing import Dict, Type
from .base_implementation import BaseImplementation

class ImplementationFactory:
    """Factory for creating implementation instances."""
    
    _implementations: Dict[str, Type[BaseImplementation]] = {}
    
    @classmethod
    def register(cls, name: str, implementation: Type[BaseImplementation]) -> None:
        """Register a new implementation.
        
        Args:
            name: Name of the implementation
            implementation: Implementation class
        """
        if not issubclass(implementation, BaseImplementation):
            raise ValueError(f"Implementation must inherit from BaseImplementation")
        cls._implementations[name] = implementation
    
    @classmethod
    def create(cls, name: str, config: Dict) -> BaseImplementation:
        """Create an instance of the specified implementation.
        
        Args:
            name: Name of the implementation to create
            config: Configuration for the implementation
            
        Returns:
            An instance of the requested implementation
            
        Raises:
            ValueError: If the implementation is not registered
        """
        if name not in cls._implementations:
            raise ValueError(f"Implementation '{name}' not found. Available implementations: {list(cls._implementations.keys())}")
        
        implementation_class = cls._implementations[name]
        return implementation_class(config) 