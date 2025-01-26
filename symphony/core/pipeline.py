from typing import List, Dict, Any, Optional
from ..embeddings import Embedder
from ..discovery import Discovery
from ..execution import Executor
from ..execution.aggregator import Aggregator
from ..decomposition import Decomposer
from pathlib import Path

class Pipeline:
    def __init__(self, index_path: Optional[Path] = None, openai_api_key: Optional[str] = None):
        """
        Initialize the pipeline components.
        
        Args:
            index_path: Optional path to load an existing discovery index
            openai_api_key: Optional OpenAI API key for decomposition and aggregation
        """
        self.embedder = Embedder()
        self.discovery = Discovery(embedder=self.embedder, index_path=index_path)
        self.executor = Executor()
        self.decomposer = Decomposer(api_key=openai_api_key)
        self.aggregator = Aggregator(api_key=openai_api_key)

    def run_query(self, query: str) -> Dict[str, Any]:
        """
        Run the complete pipeline on a given query.
        
        Args:
            query: The natural language query string
            
        Returns:
            Dict containing the answer and metadata
        """
        # 1. Discovery phase - find relevant items
        relevant_items = self._discover(query)
        
        if not relevant_items:
            return {
                "answer": "I could not find any relevant information to answer your question.",
                "confidence": 0.0,
                "source_type": None,
                "source": None
            }
        
        # 2. Decomposition phase - break down complex queries
        decomposition = self.decomposer.decompose_query(query, relevant_items)
        
        if not decomposition["requires_decomposition"]:
            # Simple query - just execute on the most relevant item
            return self._execute(query, relevant_items[0])
            
        # 3. Execute each sub-query
        sub_results = []
        for sub_query in decomposition["sub_queries"]:
            target_idx = sub_query["target_item_index"] - 1  # Convert to 0-based index
            if target_idx < len(relevant_items):
                result = self._execute(
                    sub_query["sub_query"],
                    relevant_items[target_idx]
                )
                sub_results.append(result)
                
        if not sub_results:
            return {
                "answer": "Failed to execute any sub-queries successfully.",
                "confidence": 0.0,
                "source_type": None,
                "source": None
            }
            
        # 4. Aggregate results if needed
        if len(sub_results) == 1:
            return sub_results[0]
            
        return self.aggregator.aggregate_results(
            original_query=query,
            sub_queries=decomposition["sub_queries"],
            sub_results=sub_results,
            aggregation_strategy=decomposition["aggregation_strategy"]
        )

    def _discover(self, query: str) -> List[Dict[str, Any]]:
        """Use the discovery module to find relevant items."""
        return self.discovery.discover(query, k=5, min_score=0.5)

    def _execute(self, query: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the query on a single item."""
        return self.executor.execute_query(query, item)
        
    def index_data(self, items: List[Dict[str, Any]], texts: List[str]):
        """
        Index new data items for discovery.
        
        Args:
            items: List of items to index
            texts: Text representation for each item
        """
        self.discovery.index_items(items, texts)