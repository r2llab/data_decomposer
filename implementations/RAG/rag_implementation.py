from pathlib import Path
from typing import Any, Dict, Optional, List
import pandas as pd
from difflib import SequenceMatcher

from core.base_implementation import BaseImplementation
from implementations.symphony.discovery import Discovery
from implementations.symphony.execution import Executor
from implementations.symphony.utils import CostTracker
from implementations.symphony.embeddings.auto_embedder import AutoEmbedder

class RAGImplementation(BaseImplementation):
    """Retrieval-Augmented Generation implementation."""

    def initialize(self) -> None:
        """Initialize RAG resources."""
        print("Initializing RAG")
        # Configuration
        self.openai_api_key = self.config.get('openai_api_key')
        self.index_path = self.config.get('index_path')
        if self.index_path:
            self.index_path = Path(self.index_path)

        # Cost tracker
        self.cost_tracker = CostTracker()

        # Discovery module for retrieval using OpenAI embeddings
        embedder = AutoEmbedder(api_key=self.openai_api_key)
        self.discovery = Discovery(embedder=embedder, index_path=self.index_path)

        # Executor for generation
        self.executor = Executor(api_key=self.openai_api_key, cost_tracker=self.cost_tracker)

        # Source relevance tracking
        self.ground_truth_answer: Optional[str] = None
        self.source_relevance_scores: List[float] = []
        self.document_sources: List[str] = []

    def _calculate_text_similarity(self, content: Any, reference: str) -> float:
        """
        Calculate similarity between content and reference text.
        """
        if isinstance(content, pd.DataFrame):
            try:
                text = content.head(10).to_string()
                return SequenceMatcher(None, text, reference).ratio()
            except Exception:
                return 0.0
        if isinstance(content, str):
            return SequenceMatcher(None, content, reference).ratio()
        try:
            return SequenceMatcher(None, str(content), reference).ratio()
        except Exception:
            return 0.0

    def process_query(self, query: str, ground_truth_answer: Optional[str] = None) -> Dict[str, Any]:
        """Process a query using RAG: retrieve docs, generate answer, and return metadata."""
        # Reset metrics
        self.cost_tracker.reset_query_stats()
        self.ground_truth_answer = ground_truth_answer
        self.source_relevance_scores = []
        self.document_sources = []

        # Retrieve relevant documents
        k = self.config.get('k', 5)
        min_score = self.config.get('min_score', 0.25)
        keyword_boost = self.config.get('keyword_boost', True)
        relevant_items = self.discovery.discover(
            query, k=k, min_score=min_score, keyword_boost=keyword_boost
        )

        # if not relevant_items:
        #     return {
        #         'answer': "I could not find any relevant information to answer your question.",
        #         'confidence': 0.0,
        #         'source_type': None,
        #         'source': None,
        #         'document_sources': [],
        #         'cost_metrics': {
        #             'total_cost': 0.0,
        #             'total_tokens': 0,
        #             'api_calls': 0,
        #             'model_breakdown': {},
        #             'endpoint_breakdown': {}
        #         }
        #     }

        # Collect contexts and track sources and relevance
        contexts: List[str] = []
        for item in relevant_items:
            content = item.get('data') if item.get('data') is not None else item.get('content')
            if content is None:
                continue
            # Convert content to string
            if isinstance(content, pd.DataFrame):
                content_str = content.to_string()
            else:
                content_str = str(content)
            contexts.append(content_str)

            # Track document sources
            source = item.get('metadata', {}).get('source') or item.get('source')
            if source:
                self.document_sources.append(source)

            # Compute source relevance if ground truth is available
            if self.ground_truth_answer:
                rel = self._calculate_text_similarity(content_str, self.ground_truth_answer)
                self.source_relevance_scores.append(rel)

        # Build combined context
        combined_context = "\n\n---\n\n".join(contexts)

        # Create a synthetic item for combined context
        combined_item = {'type': 'text', 'data': combined_context}

        # Generate answer
        result = self.executor.execute_query(query, combined_item)

        # Attach retrieved document sources
        result['document_sources'] = self.document_sources

        # Attach cost metrics
        cost_summary = self.cost_tracker.get_query_summary()
        result['cost_metrics'] = {
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

        # Attach source relevance scores
        if self.ground_truth_answer and self.source_relevance_scores:
            avg_rel = sum(self.source_relevance_scores) / len(self.source_relevance_scores)
            max_rel = max(self.source_relevance_scores)
            result['source_relevance_score'] = {
                'average': float(avg_rel),
                'maximum': float(max_rel),
                'scores': [float(s) for s in self.source_relevance_scores]
            }

        return result

    def cleanup(self) -> None:
        """Cleanup RAG resources and print usage summary."""
        summary = self.cost_tracker.get_cost_summary()
        print("\nTotal usage summary:")
        print(f"Total cost: ${summary['total_cost']:.6f}")
        print(f"Total tokens: {summary['total_tokens']}")
        print(f"Total API calls: {summary['total_calls']}")
        print("\nModel breakdown:")
        for model, stats in summary['models'].items():
            print(f"  {model}: ${stats['cost']:.6f} ({stats['calls']} calls)")
        print("\nEndpoint breakdown:")
        for endpoint, stats in summary['endpoints'].items():
            print(f"  {endpoint}: ${stats['cost']:.6f} ({stats['calls']} calls)")