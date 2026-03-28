from core.factory import ImplementationFactory
from .rag_implementation import RAGImplementation
from .azure_rag_implementation import AzureSearchRAGImplementation

# Register RAG implementation
ImplementationFactory.register('rag', RAGImplementation)
ImplementationFactory.register('rag_azure', AzureSearchRAGImplementation)
