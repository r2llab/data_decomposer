from core.factory import ImplementationFactory
from .rag_implementation import RAGImplementation

# Register RAG implementation
ImplementationFactory.register('rag', RAGImplementation) 