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
                         aggregation_strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Aggregate results from multiple sub-queries into a final answer.
        
        Args:
            original_query: The original user query
            sub_queries: List of sub-query specifications
            sub_results: List of results from executing sub-queries
            aggregation_strategy: Optional strategy for aggregation
            
        Returns:
            Dict containing aggregated answer and metadata
        """
        # Create context from sub-results
        context = self._create_aggregation_context(original_query, sub_queries, sub_results)
        
        # Get aggregated answer from GPT
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a precise answer aggregator. Combine the results of sub-queries into a complete answer."},
                {"role": "user", "content": context}
            ],
            temperature=0.3
        )
        
        # Calculate confidence based on sub-results
        confidence = sum(result.get("confidence", 0.0) for result in sub_results) / len(sub_results)
        
        # Collect all source types and sources
        source_types = list(set(result.get("source_type") for result in sub_results if result.get("source_type")))
        sources = [result.get("source") for result in sub_results if result.get("source")]
        
        return {
            "answer": response.choices[0].message.content,
            "confidence": confidence,
            "source_type": source_types if source_types else None,
            "source": sources if sources else None,
            "sub_results": sub_results  # Include sub-results for reference
        }
        
    def _create_aggregation_context(self,
                                  query: str,
                                  sub_queries: List[Dict[str, Any]],
                                  sub_results: List[Dict[str, Any]]) -> str:
        """Create the context for GPT aggregation."""
        context_parts = [
            f"Original Query: {query}\n",
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