from typing import Dict, Any, List, Optional
import openai
from ..utils.prompts import create_chat_completion, init_openai
from ..utils.cost_tracker import CostTracker

class LLMGenerator:
    """LLM-based implementation of the ReSP Generator component"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", cost_tracker: Optional[CostTracker] = None):
        """
        Initialize the LLM Generator.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for generation
            cost_tracker: Optional cost tracker for tracking API usage
        """
        self.api_key = api_key
        self.model = model
        self.cost_tracker = cost_tracker
        init_openai(api_key)
        
    def generate(self,
                question: str,
                global_evidence: List[str],
                local_pathway: List[str],
                document_sources: List[str] = None) -> Dict[str, Any]:
        """
        Generate final answer using LLM.
        
        Args:
            question: The original query (Q)
            global_evidence: List of summaries relevant to main question
            local_pathway: List of summaries from the reasoning pathway
            document_sources: List of document/table IDs used in the process
            
        Returns:
            Dict containing answer, confidence, supporting evidence, and document sources
        """
        # Construct prompt with all available evidence
        prompt = self._construct_generation_prompt(
            question=question,
            global_evidence=global_evidence,
            local_pathway=local_pathway,
            document_sources=document_sources
        )
        
        # Get LLM response
        response = create_chat_completion(
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            cost_tracker=self.cost_tracker,
            metadata={"component": "generator", "type": "final_answer", "question": question}
        )
        
        # Parse response
        result = self._parse_generation_response(response)
        
        # Add document sources to the result
        if document_sources:
            result["document_sources"] = document_sources
        
        return result
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for generation"""
        return """You are an expert question answerer in a recursive evidence-based question answering system.
Your task is to generate a final answer to a user question based on the provided evidence.
Follow the requested output format precisely.
Be faithful to the evidence - do not make up information not supported by the evidence.
Include ONLY information that is directly supported by the provided evidence.
"""
    
    def _construct_generation_prompt(self,
                                   question: str,
                                   global_evidence: List[str],
                                   local_pathway: List[str],
                                   document_sources: List[str] = None) -> str:
        """
        Construct prompt for final answer generation.
        
        Args:
            question: Original question
            global_evidence: Evidence summaries for main question
            local_pathway: Step-by-step summaries from reasoning
            document_sources: List of document/table IDs used
        """
        # Build evidence sections
        global_evidence_text = "\n\n".join([f"- {ev}" for ev in global_evidence])
        local_pathway_text = "\n\n".join([f"- {step}" for step in local_pathway])
        
        # Include document sources information
        document_sources_text = ""
        if document_sources:
            formatted_sources = []
            for src in document_sources:
                if src and src != "unknown":
                    formatted_sources.append(f"- {src}")
            
            if formatted_sources:
                document_sources_text = "\n\nDocuments/Tables Used:\n" + "\n".join(formatted_sources)
        
        return f"""Question: {question}

Global Evidence:
{global_evidence_text}

Reasoning Pathway:
{local_pathway_text}
{document_sources_text}

Based on the provided evidence and reasoning pathway, please generate a final answer to the question.
Your response should follow this structure:

ANSWER: [Your concise answer]

CONFIDENCE: [A number between 0.0 and 1.0 indicating your confidence in the answer]

EVIDENCE: 
- [Key piece of evidence supporting the answer]
- [Another key piece of evidence]
...

DOCUMENT SOURCES:
- [List each document/table source that was used]
- [e.g., Target-12345]
- [e.g., drugbank-drug.csv]

IMPORTANT: Include ALL document sources that were used to generate this answer. 
This should include Target documents (Target-XXXXX) and any drugbank tables.
"""
    
    def _parse_generation_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured output"""
        lines = response.strip().split("\n")
        result = {
            "answer": "",
            "confidence": 0.0,
            "supporting_evidence": [],
            "document_sources": []
        }
        
        current_section = None
        answer_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("ANSWER:"):
                current_section = "answer"
                answer_lines.append(line.split(":", 1)[1].strip())
            elif line.startswith("CONFIDENCE:"):
                current_section = "confidence"
                try:
                    result["confidence"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    result["confidence"] = 0.0
            elif line.startswith("EVIDENCE:"):
                current_section = "evidence"
                # Join accumulated answer lines when moving to another section
                if answer_lines:
                    result["answer"] = " ".join(answer_lines)
            elif line.startswith("DOCUMENT SOURCES:") or line.startswith("DOCUMENTS/TABLES USED:") or line.startswith("SOURCES:"):
                current_section = "document_sources"
                # Join accumulated answer lines when moving to another section
                if answer_lines and not result["answer"]:
                    result["answer"] = " ".join(answer_lines)
            elif current_section == "answer" and not line.startswith("-") and not any(line.startswith(prefix) for prefix in ["ANSWER:", "CONFIDENCE:", "EVIDENCE:", "DOCUMENT SOURCES:", "DOCUMENTS/TABLES USED:", "SOURCES:"]):
                # Capture additional answer lines as long as they don't start a new section
                answer_lines.append(line)
            elif current_section == "evidence" and line.startswith("-"):
                result["supporting_evidence"].append(line.strip()[2:].strip())
            elif current_section == "document_sources" and line.startswith("-"):
                source = line.strip()[2:].strip()
                if source:  # Only add non-empty sources
                    result["document_sources"].append(source)
        
        # Make sure to include answer if we haven't yet
        if answer_lines and not result["answer"]:
            result["answer"] = " ".join(answer_lines)
                
        return result 