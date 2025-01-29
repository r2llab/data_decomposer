from typing import Dict, Any, List, Protocol

class Generator(Protocol):
    """Protocol defining the interface for the ReSP Generator component"""
    
    def generate(self,
                question: str,
                global_evidence: List[str],
                local_pathway: List[str]) -> Dict[str, Any]:
        """
        Generate final answer based on accumulated evidence.
        
        Args:
            question: The original query (Q)
            global_evidence: List of summaries relevant to main question
            local_pathway: List of summaries from the reasoning pathway
            
        Returns:
            Dict containing:
                - answer: The generated answer
                - confidence: Confidence score
                - supporting_evidence: List of evidence used
        """
        ... 