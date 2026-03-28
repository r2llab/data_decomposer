from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict, List, Optional
import time


@dataclass
class ApiCall:
    """Represents a single API call with its metadata and cost."""

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
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))


class CostTracker:
    """Tracks API usage/cost at total and per-query levels."""

    MODEL_PRICES = {
        "gpt-4o": (0.01, 0.03),
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-4-turbo-preview": (0.01, 0.03),
        "gpt-3.5-turbo": (0.0005, 0.0015),
        "text-embedding-3-small": (0.00002, 0.0),
        "text-embedding-3-large": (0.00013, 0.0),
        "text-embedding-ada-002": (0.0001, 0.0),
    }

    def __init__(self) -> None:
        self.calls: List[ApiCall] = []
        self.reset_query_stats()

    def reset_query_stats(self) -> None:
        self.query_calls: List[ApiCall] = []

    def track_embedding_call(
        self, model: str, input_count: int, embedding_dimensions: int = 1536
    ) -> ApiCall:
        del embedding_dimensions
        estimated_tokens = input_count * 100
        input_price, _ = self.MODEL_PRICES.get(model, (0.0001, 0.0))
        cost = (estimated_tokens / 1000) * input_price

        call = ApiCall(
            timestamp=time.time(),
            model=model,
            endpoint="embedding",
            prompt_tokens=estimated_tokens,
            total_tokens=estimated_tokens,
            cost=cost,
            metadata={"input_count": input_count},
        )
        self.calls.append(call)
        self.query_calls.append(call)
        return call

    def track_chat_completion_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApiCall:
        input_price, output_price = self.MODEL_PRICES.get(model, (0.01, 0.03))
        prompt_cost = (prompt_tokens / 1000) * input_price
        completion_cost = (completion_tokens / 1000) * output_price

        call = ApiCall(
            timestamp=time.time(),
            model=model,
            endpoint="chat.completions",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=prompt_cost + completion_cost,
            metadata=metadata or {},
        )
        self.calls.append(call)
        self.query_calls.append(call)
        return call

    def _summary_for_calls(self, calls: List[ApiCall]) -> Dict[str, Any]:
        summary = {
            "cost": float(sum(call.cost for call in calls)),
            "tokens": int(sum(call.total_tokens for call in calls)),
            "calls": int(len(calls)),
            "models": {},
            "endpoints": {},
        }

        for call in calls:
            model_stats = summary["models"].setdefault(
                call.model, {"cost": 0.0, "tokens": 0, "calls": 0}
            )
            model_stats["cost"] += float(call.cost)
            model_stats["tokens"] += int(call.total_tokens)
            model_stats["calls"] += 1

            endpoint_stats = summary["endpoints"].setdefault(
                call.endpoint, {"cost": 0.0, "tokens": 0, "calls": 0}
            )
            endpoint_stats["cost"] += float(call.cost)
            endpoint_stats["tokens"] += int(call.total_tokens)
            endpoint_stats["calls"] += 1

        return summary

    def get_cost_summary(self) -> Dict[str, Any]:
        base = self._summary_for_calls(self.calls)
        return {
            "total_cost": base["cost"],
            "total_tokens": base["tokens"],
            "total_calls": base["calls"],
            "models": base["models"],
            "endpoints": base["endpoints"],
        }

    def get_query_summary(self) -> Dict[str, Any]:
        base = self._summary_for_calls(self.query_calls)
        return {
            "query_cost": base["cost"],
            "query_tokens": base["tokens"],
            "query_calls": base["calls"],
            "models": base["models"],
            "endpoints": base["endpoints"],
        }
