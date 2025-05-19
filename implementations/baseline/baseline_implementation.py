from typing import Any, Dict, Optional
import time
import openai
from dataclasses import dataclass, field
from core.base_implementation import BaseImplementation


@dataclass
class ApiCall:
    """Represents a single API call with its associated metadata and cost."""
    timestamp: float
    model: str
    endpoint: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def formatted_timestamp(self) -> str:
        """Return a human-readable timestamp."""
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))


class CostTracker:
    """
    Utility class to track API calls and their associated costs.
    Simplified version that only tracks chat completion calls.
    """
    
    # Current pricing per 1K tokens as of 2023 (update as pricing changes)
    # Format: (input_price_per_1k, output_price_per_1k)
    MODEL_PRICES = {
        # GPT-4o
        "gpt-4o": (0.01, 0.03),
    }
    
    def __init__(self):
        """Initialize the cost tracker."""
        self.calls = []
        self.reset_query_stats()
    
    def reset_query_stats(self):
        """Reset per-query statistics."""
        self.query_calls = []
    
    def track_chat_completion_call(self, model: str, prompt_tokens: int, completion_tokens: int, metadata: Optional[Dict[str, Any]] = None):
        """Track a chat completion API call."""
        # Get pricing for this model
        input_price, output_price = self.MODEL_PRICES.get(model, (0.01, 0.03))
        
        # Calculate cost
        prompt_cost = (prompt_tokens / 1000) * input_price
        completion_cost = (completion_tokens / 1000) * output_price
        total_cost = prompt_cost + completion_cost
        
        # Create and store the call record
        call = ApiCall(
            timestamp=time.time(),
            model=model,
            endpoint="chat.completions",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=total_cost,
            metadata=metadata or {}
        )
        
        self.calls.append(call)
        self.query_calls.append(call)
        
        return call
    
    def get_query_summary(self) -> Dict[str, Any]:
        """Get a summary of costs for the current query."""
        query_cost = sum(call.cost for call in self.query_calls)
        query_tokens = sum(call.total_tokens for call in self.query_calls)
        query_calls = len(self.query_calls)
        
        summary = {
            "query_cost": float(query_cost),
            "query_tokens": int(query_tokens),
            "query_calls": int(query_calls),
            "models": {},
            "endpoints": {}
        }
        
        # Group by model
        for call in self.query_calls:
            if call.model not in summary["models"]:
                summary["models"][call.model] = {
                    "cost": 0.0,
                    "tokens": 0,
                    "calls": 0
                }
            summary["models"][call.model]["cost"] += float(call.cost)
            summary["models"][call.model]["tokens"] += int(call.total_tokens)
            summary["models"][call.model]["calls"] += 1
            
            # Group by endpoint
            if call.endpoint not in summary["endpoints"]:
                summary["endpoints"][call.endpoint] = {
                    "cost": 0.0,
                    "tokens": 0,
                    "calls": 0
                }
            summary["endpoints"][call.endpoint]["cost"] += float(call.cost)
            summary["endpoints"][call.endpoint]["tokens"] += int(call.total_tokens)
            summary["endpoints"][call.endpoint]["calls"] += 1
        
        return summary


class BaselineImplementation(BaseImplementation):
    """
    Baseline implementation of the BaseImplementation interface.
    
    This implementation simply sends the user's question directly to GPT-4o
    without any additional context or processing.
    """
    
    def initialize(self) -> None:
        """Initialize the baseline implementation resources."""
        print("Initializing Baseline Implementation")
        
        # Extract config values
        self.openai_api_key = self.config.get('openai_api_key')
        if not self.openai_api_key:
            raise ValueError("openai_api_key is required in config")
        
        # Set up OpenAI client
        self.client = openai.OpenAI(api_key=self.openai_api_key)
        
        # Create cost tracker
        self.cost_tracker = CostTracker()
    
    def process_query(self, query: str, ground_truth_answer: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a query by sending it directly to GPT-4o.
        
        Args:
            query: The query string to process
            ground_truth_answer: Optional ground truth answer (not used)
            
        Returns:
            Dict containing the answer and metadata
        """
        print(f"Running query: {query}")
        
        # Reset query-specific cost tracking
        self.cost_tracker.reset_query_stats()
        
        # Make a direct call to GPT-4o with the query
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Provide direct answers to questions without additional explanations or information. Be concise and to the point."},
                {"role": "user", "content": query}
            ]
        )
        
        # Extract the answer
        answer = response.choices[0].message.content
        
        # Track the API call for cost calculation
        self.cost_tracker.track_chat_completion_call(
            model="gpt-4o",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )
        
        # Get cost metrics
        cost_summary = self.cost_tracker.get_query_summary()
        
        # Build the result dictionary
        result = {
            "answer": answer,
            "document_sources": [],  # Empty as we don't use any sources
            "cost_metrics": {
                'total_cost': float(cost_summary['query_cost']),
                'total_tokens': int(cost_summary['query_tokens']),
                'api_calls': int(cost_summary['query_calls']),
                'model_breakdown': {
                    model: {
                        'cost': float(stats['cost']),
                        'tokens': int(stats['tokens']),
                        'calls': int(stats['calls'])
                    }
                    for model, stats in cost_summary['models'].items()
                },
                'endpoint_breakdown': {
                    endpoint: {
                        'cost': float(stats['cost']),
                        'tokens': int(stats['tokens']),
                        'calls': int(stats['calls'])
                    }
                    for endpoint, stats in cost_summary['endpoints'].items()
                }
            }
        }
        
        print(f"Query cost: ${cost_summary['query_cost']:.6f}")
        print(f"Total tokens: {cost_summary['query_tokens']}")
        print(f"API calls: {cost_summary['query_calls']}")
        
        return result
    
    def cleanup(self) -> None:
        """Cleanup any resources used by the implementation."""
        # Nothing to clean up for the baseline implementation
        pass
