from typing import Dict, Any, List
import openai
from ..utils.prompts import create_chat_completion, init_openai

class LLMGenerator:
    """LLM-based implementation of the ReSP Generator component"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the LLM Generator.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for generation
        """
        self.api_key = api_key
        self.model = model
        init_openai(api_key)
        
    def generate(self,
                question: str,
                global_evidence: List[str],
                local_pathway: List[str]) -> Dict[str, Any]:
        """
        Generate final answer using LLM.
        
        Args:
            question: The original query (Q)
            global_evidence: List of summaries relevant to main question
            local_pathway: List of summaries from the reasoning pathway
            
        Returns:
            Dict containing answer, confidence, and supporting evidence
        """
        # Construct prompt with all available evidence
        prompt = self._construct_generation_prompt(
            question=question,
            global_evidence=global_evidence,
            local_pathway=local_pathway
        )
        
        # Get LLM response
        response = create_chat_completion(
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            model=self.model
        )
        
        # Parse response
        return self._parse_generation_response(response)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for generation"""
        return """You are an expert answer generator in a recursive question-answering system.
Your task is to generate a comprehensive and accurate answer based on the provided evidence.

Respond in the following format:
ANSWER: <your detailed answer>
CONFIDENCE: <confidence score between 0 and 1>
EVIDENCE: <list of key evidence pieces used>"""
    
    def _construct_generation_prompt(self,
                                   question: str,
                                   global_evidence: List[str],
                                   local_pathway: List[str]) -> str:
        """Construct the generation prompt"""
        return f"""Question: {question}

Global Evidence (relevant to main question):
{self._format_evidence(global_evidence)}

Local Pathway Evidence (from reasoning process):
{self._format_evidence(local_pathway)}

Based on the above evidence, please generate a comprehensive answer to the question.
Ensure your answer is well-supported by the evidence and indicate your confidence level.
If there are any uncertainties or gaps in the evidence, acknowledge them in your response."""
    
    def _format_evidence(self, evidence: List[str]) -> str:
        """Format evidence list for prompt"""
        return "\n".join(f"- {item}" for item in evidence)
    
    def _parse_generation_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured output"""
        lines = response.strip().split("\n")
        result = {
            "answer": "",
            "confidence": 0.0,
            "supporting_evidence": []
        }
        
        current_section = None
        for line in lines:
            if line.startswith("ANSWER:"):
                current_section = "answer"
                result["answer"] = line.split(":", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    result["confidence"] = 0.0
            elif line.startswith("EVIDENCE:"):
                current_section = "evidence"
            elif current_section == "evidence" and line.strip().startswith("-"):
                result["supporting_evidence"].append(line.strip()[2:].strip())
                
        return result 