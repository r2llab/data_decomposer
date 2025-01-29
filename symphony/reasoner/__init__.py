from typing import Dict, Any, List, Protocol

class Reasoner(Protocol):
    """Protocol defining the interface for the ReSP Reasoner component"""
    
    def reason(self,
               main_question: str,
               current_sub_question: str,
               global_evidence: List[str],
               local_pathway: List[str]) -> Dict[str, Any]:
        """
        Decide whether to continue iteration with a new sub-question or exit.
        
        Args:
            main_question: The original query (Q)
            current_sub_question: The current sub-question being processed (Q*)
            global_evidence: List of summaries relevant to main question
            local_pathway: List of summaries relevant to current sub-question
            
        Returns:
            Dict containing:
                - should_exit (bool): Whether to exit iteration
                - next_sub_question (str, optional): Next sub-question if continuing
                - reasoning (str, optional): Explanation of the decision
        """
        ... 