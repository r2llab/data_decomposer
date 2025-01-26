from typing import List, Dict, Any, Optional
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from dotenv import load_dotenv

load_dotenv()

class Aggregator:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the aggregator with OpenAI API.
        
        Args:
            api_key: OpenAI API key. If None, loads from environment variable.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
            
        self.client = openai.OpenAI(api_key=self.api_key)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def aggregate_results(self,
                         original_query: str,
                         sub_queries: List[Dict[str, Any]],
                         sub_results: List[Dict[str, Any]],
                         aggregation_strategy: str) -> Dict[str, Any]:
        """
        Aggregate results from multiple sub-queries into a final answer.
        
        Args:
            original_query: The original user query
            sub_queries: List of sub-query specifications
            sub_results: List of results from executing sub-queries
            aggregation_strategy: Strategy for combining results
            
        Returns:
            Dict containing the final answer and metadata
        """
        # Create context for GPT
        context = self._create_aggregation_context(
            original_query, sub_queries, sub_results, aggregation_strategy
        )
        
        # Get aggregation from GPT
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are an expert at combining information 
                from multiple sources to create coherent answers. Analyze the sub-query results 
                and create a comprehensive answer."""},
                {"role": "user", "content": context}
            ],
            temperature=0.3
        )
        
        aggregated_answer = response.choices[0].message.content.strip()
        
        # Calculate confidence as average of sub-results' confidences
        confidence = sum(r.get("confidence", 0) for r in sub_results) / len(sub_results)
        
        return {
            "answer": aggregated_answer,
            "confidence": confidence,
            "sub_results": sub_results,
            "aggregation_strategy": aggregation_strategy
        }
        
    def _create_aggregation_context(self,
                                  query: str,
                                  sub_queries: List[Dict[str, Any]],
                                  sub_results: List[Dict[str, Any]],
                                  strategy: str) -> str:
        """Create the context for GPT aggregation."""
        context_parts = [
            f"Original Query: {query}\n",
            f"Aggregation Strategy: {strategy}\n",
            "\nSub-queries and their results:"
        ]
        
        for sq, sr in zip(sub_queries, sub_results):
            context_parts.append(
                f"\nSub-query: {sq['sub_query']}\n"
                f"Answer: {sr.get('answer', 'No answer')}\n"
                f"Confidence: {sr.get('confidence', 0)}"
            )
            
        context_parts.append("\nPlease combine these results into a coherent answer to the original query.")
        
        return "\n".join(context_parts) 