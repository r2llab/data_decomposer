from typing import Dict, Any, List, Optional
import openai
from ..utils.prompts import create_chat_completion, init_openai
from ..utils.cost_tracker import CostTracker

class LLMReasoner:
    """LLM-based implementation of the ReSP Reasoner component"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", cost_tracker: Optional[CostTracker] = None):
        """
        Initialize the LLM Reasoner.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for reasoning
            cost_tracker: Optional cost tracker for tracking API usage
        """
        self.api_key = api_key
        self.model = model
        self.cost_tracker = cost_tracker
        init_openai(api_key)
        
    def reason(self,
               main_question: str,
               current_sub_question: str,
               global_evidence: List[str],
               local_pathway: List[str]) -> Dict[str, Any]:
        """
        Use LLM to decide whether to continue iteration with a new sub-question or exit.
        
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
        # Construct prompt
        prompt = self._construct_reasoning_prompt(
            main_question=main_question,
            current_sub_question=current_sub_question,
            global_evidence=global_evidence,
            local_pathway=local_pathway
        )
        
        # Get LLM response
        response = create_chat_completion(
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            cost_tracker=self.cost_tracker,
            metadata={"component": "reasoner", "type": "reasoning", "main_question": main_question}
        )
        
        # Parse response
        return self._parse_reasoning_response(response)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the reasoning task"""
        return """You are an expert reasoner in a recursive question-answering system.
Your task is to:
1. Analyze the current evidence and determine if enough information has been gathered
2. If more information is needed, generate a focused sub-question
3. If sufficient information exists, signal to exit the iteration

Respond in the following format:
SHOULD_EXIT: true/false
REASONING: <your reasoning>
NEXT_QUESTION: <next sub-question if should_exit is false>"""
    
    def _construct_reasoning_prompt(self,
                                  main_question: str,
                                  current_sub_question: str,
                                  global_evidence: List[str],
                                  local_pathway: List[str]) -> str:
        """Construct the reasoning prompt"""
        return f"""Main Question: {main_question}
Current Sub-question: {current_sub_question}

Global Evidence (relevant to main question):
{self._format_evidence(global_evidence)}

Local Pathway (relevant to current sub-question):
{self._format_evidence(local_pathway)}

Based on the above information, should we:
1. Exit iteration because we have sufficient information to answer the main question
2. Continue with a new sub-question to gather more specific information

Please provide your reasoning and decision."""
    
    def _format_evidence(self, evidence: List[str]) -> str:
        """Format evidence list for prompt"""
        return "\n".join(f"- {item}" for item in evidence)
    
    def _parse_reasoning_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured output"""
        lines = response.strip().split("\n")
        result = {
            "should_exit": False,
            "reasoning": "",
            "next_sub_question": None
        }
        
        for line in lines:
            if line.startswith("SHOULD_EXIT:"):
                result["should_exit"] = line.split(":", 1)[1].strip().lower() == "true"
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.split(":", 1)[1].strip()
            elif line.startswith("NEXT_QUESTION:"):
                result["next_sub_question"] = line.split(":", 1)[1].strip()
        
        return result 