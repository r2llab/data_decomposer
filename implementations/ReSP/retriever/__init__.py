from typing import Dict, Any, List, Protocol

class Retriever(Protocol):
    """Protocol defining the interface for the ReSP Retriever component"""
    
    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on the question.
        
        Args:
            question: The question to search for (either Q or Q*)
            
        Returns:
            List of documents, each containing:
                - content: The document content
                - metadata: Any relevant metadata (source, score, etc.)
        """
        ... 