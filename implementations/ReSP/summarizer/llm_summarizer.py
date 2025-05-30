from typing import Dict, Any, List, Optional
import openai
from ..utils.prompts import create_chat_completion, init_openai
from ..utils.cost_tracker import CostTracker
import pandas as pd

class LLMSummarizer:
    """LLM-based implementation of the ReSP Summarizer component"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", cost_tracker: Optional[CostTracker] = None):
        """
        Initialize the LLM Summarizer.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for summarization
            cost_tracker: Optional cost tracker for tracking API usage
        """
        self.api_key = api_key
        self.model = model
        self.cost_tracker = cost_tracker
        init_openai(api_key)
        
    def summarize(self,
                 documents: List[Dict[str, Any]],
                 main_question: str,
                 sub_question: str) -> Dict[str, str]:
        """
        Perform dual summarization using LLM.
        
        Args:
            documents: List of retrieved documents to summarize
            main_question: The original query (Q)
            sub_question: The current sub-question (Q*)
            
        Returns:
            Dict containing global and local summaries
        """
        # Prepare document content


        doc_texts = [self._format_document(doc) for doc in documents]
        combined_docs = "\n\n".join(doc_texts)


        # Log the combined documents
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Combined documents for summarization:\n{combined_docs}")
        
        # Get global summary (for Q)
        global_summary = self._get_focused_summary(
            documents=combined_docs,
            question=main_question,
            summary_type="global"
        )
        
        # Get local summary (for Q*)
        local_summary = self._get_focused_summary(
            documents=combined_docs,
            question=sub_question,
            summary_type="local"
        )
        
        return {
            "global_summary": global_summary,
            "local_summary": local_summary
        }
    
    def _format_document(self, doc: Dict[str, Any]) -> str:
        """Format a document for inclusion in prompt"""
        content = doc["content"]
        metadata = doc["metadata"]
        
        # Handle DataFrame content specially to avoid truncation
        if isinstance(content, pd.DataFrame):
            # Convert DataFrame to a more readable format without truncation
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
                # Convert to JSON for a more compact representation that preserves all data
                df_json = content.to_json(orient='records')
                content_str = f"DataFrame with {content.shape[0]} rows and {content.shape[1]} columns.\nColumns: {', '.join(content.columns)}\nData: {df_json}"
        else:
            content_str = str(content)
        
        return f"""Source: {metadata.get('source', 'unknown')}
Type: {metadata.get('type', 'unknown')}
Relevance: {metadata.get('score', 0.0)}
Content: {content_str}"""
    
    def _get_focused_summary(self,
                           documents: str,
                           question: str,
                           summary_type: str) -> str:
        """Get a focused summary for a specific question"""
        prompt = self._construct_summary_prompt(
            documents=documents,
            question=question,
            summary_type=summary_type
        )

        
        response = create_chat_completion(
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            cost_tracker=self.cost_tracker,
            metadata={"component": "summarizer", "type": summary_type, "question": question}
        )
        
        return response.strip()
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for summarization"""
        return """You are an expert summarizer in a recursive question-answering system.
Your task is to create focused summaries of documents that extract information relevant to specific questions.
Keep summaries concise but include all relevant details that could help answer the question."""
    
    def _construct_summary_prompt(self,
                                documents: str,
                                question: str,
                                summary_type: str) -> str:
        """Construct the summarization prompt"""
        context = "main question" if summary_type == "global" else "current sub-question"
        
        return f"""Question ({context}): {question}

Documents to summarize:
{documents}

Please provide a focused summary that extracts and synthesizes information from these documents
that is specifically relevant to answering the above question. Include only information that
could help answer this specific question.""" 