from typing import Dict, Any, List, Protocol

class Summarizer(Protocol):
    """Protocol defining the interface for the ReSP Summarizer component"""
    
    def summarize(self,
                 documents: List[Dict[str, Any]],
                 main_question: str,
                 sub_question: str) -> Dict[str, str]:
        """
        Perform dual summarization of documents for both Q and Q*.
        
        Args:
            documents: List of retrieved documents to summarize
            main_question: The original query (Q)
            sub_question: The current sub-question (Q*)
            
        Returns:
            Dict containing:
                - global_summary: Summary relevant to main question
                - local_summary: Summary relevant to sub-question
        """
        ... 