from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass, field


@dataclass
class ApiCall:
    """Represents a single API call with its associated metadata and cost."""
    timestamp: float
    model: str
    endpoint: str  # 'embedding', 'chat', etc.
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
    
    This tracks OpenAI API calls and calculates costs based on model pricing.
    """
    
    # Current pricing per 1K tokens as of Apr 2023 (should be updated as pricing changes)
    # Format: (input_price_per_1k, output_price_per_1k)
    MODEL_PRICES = {
        # GPT-4o
        "gpt-4o": (0.01, 0.03),
        # GPT-4 Turbo
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-4-turbo-preview": (0.01, 0.03),
        # GPT-3.5 Turbo
        "gpt-3.5-turbo": (0.0005, 0.0015),
        # Embedding models
        "text-embedding-3-small": (0.00002, 0.0),
        "text-embedding-3-large": (0.00013, 0.0),
        "text-embedding-ada-002": (0.0001, 0.0),
    }
    
    def __init__(self):
        """Initialize the cost tracker."""
        self.calls: List[ApiCall] = []
        self.reset_query_stats()
    
    def reset_query_stats(self):
        """Reset per-query statistics."""
        self.query_calls: List[ApiCall] = []
    
    def track_embedding_call(self, model: str, input_count: int, embedding_dimensions: int = 1536):
        """Track an embedding model API call.
        
        Args:
            model: The name of the embedding model
            input_count: Number of text chunks embedded
            embedding_dimensions: Dimensions of the embeddings
        """
        # Estimate token count based on typical usage (rough estimate)
        # A rough heuristic: ~100 tokens per text chunk on average
        estimated_tokens = input_count * 100
        
        # Get pricing for this model
        input_price, _ = self.MODEL_PRICES.get(model, (0.0001, 0.0))
        
        # Calculate cost
        cost = (estimated_tokens / 1000) * input_price
        
        # Create and store the call record
        call = ApiCall(
            timestamp=time.time(),
            model=model,
            endpoint="embedding",
            prompt_tokens=estimated_tokens,
            total_tokens=estimated_tokens,
            cost=cost,
            metadata={"input_count": input_count}
        )
        
        self.calls.append(call)
        self.query_calls.append(call)
        
        return call
    
    def track_chat_completion_call(self, model: str, prompt_tokens: int, completion_tokens: int, metadata: Optional[Dict[str, Any]] = None):
        """Track a chat completion API call.
        
        Args:
            model: The model used (e.g., 'gpt-4o')
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            metadata: Optional metadata about the call
        """
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
    
    def get_total_cost(self) -> float:
        """Get the total cost of all tracked API calls."""
        return sum(call.cost for call in self.calls)
    
    def get_query_cost(self) -> float:
        """Get the cost of the current query."""
        return sum(call.cost for call in self.query_calls)
    
    def get_total_tokens(self) -> int:
        """Get the total number of tokens used."""
        return sum(call.total_tokens for call in self.calls)
    
    def get_query_tokens(self) -> int:
        """Get the number of tokens used in the current query."""
        return sum(call.total_tokens for call in self.query_calls)
    
    def get_call_count(self) -> int:
        """Get the total number of API calls made."""
        return len(self.calls)
    
    def get_query_call_count(self) -> int:
        """Get the number of API calls made for the current query."""
        return len(self.query_calls)
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get a summary of costs grouped by model and endpoint."""
        summary = {
            "total_cost": float(self.get_total_cost()),
            "total_tokens": int(self.get_total_tokens()),
            "total_calls": int(self.get_call_count()),
            "models": {},
            "endpoints": {}
        }
        
        # Group by model
        for call in self.calls:
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
    
    def get_query_summary(self) -> Dict[str, Any]:
        """Get a summary of costs for the current query."""
        summary = {
            "query_cost": float(self.get_query_cost()),
            "query_tokens": int(self.get_query_tokens()),
            "query_calls": int(self.get_query_call_count()),
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