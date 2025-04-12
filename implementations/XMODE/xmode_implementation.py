from pathlib import Path
from typing import Any, Dict, Optional
from core.base_implementation import BaseImplementation
from .core import Pipeline
from .utils.cost_tracker import CostTracker

class XMODEImplementation(BaseImplementation):
    """XMODE implementation of the BaseImplementation interface."""
    
    def initialize(self) -> None:
        """Initialize XMODE resources."""
        print("Initializing XMODE")
        self.openai_api_key = self.config.get('openai_api_key')
        self.index_path = self.config.get('index_path')
        self.langchain_api_key = self.config.get('langchain_api_key')
        
        # Get passage retrieval paths if provided
        self.passage_embeddings_dir = self.config.get('passage_embeddings_dir')
        self.passage_index_dir = self.config.get('passage_index_dir')
        
        # Create cost tracker
        self.cost_tracker = CostTracker()
        
        if not self.openai_api_key:
            raise ValueError("openai_api_key is required in config")
        if not self.index_path:
            raise ValueError("index_path is required in config")
        
        # If passage paths not explicitly set, try to use default locations
        if not self.passage_embeddings_dir or not self.passage_index_dir:
            # Default locations in data directory
            data_dir = Path(self.index_path).parent
            self.passage_embeddings_dir = data_dir / "passage-embeddings"
            self.passage_index_dir = data_dir / "passage-index"
            
        # Initialize the pipeline with all necessary configurations    
        self.pipeline = Pipeline(
            index_path=Path(self.index_path),
            openai_api_key=self.openai_api_key,
            langchain_api_key=self.langchain_api_key,
            cost_tracker=self.cost_tracker
        )
    
    def process_query(self, query: str, ground_truth_answer: Optional[str] = None) -> Dict[str, Any]:
        """Process a query using XMODE.
        
        Args:
            query: The query string to process
            ground_truth_answer: Optional ground truth answer for relevance scoring
            
        Returns:
            Dict containing the answer and metadata
        """
        print(f"Running query: {query}")
        
        # Reset query-specific cost tracking
        self.cost_tracker.reset_query_stats()
        
        result = self.pipeline.run_query(query)
        
        # Ensure required fields exist
        if "answer" not in result:
            result["answer"] = "No answer found"
        
        if "document_sources" not in result:
            result["document_sources"] = []
        
        # Check if cost metrics are already included (from pipeline)
        if "cost_metrics" not in result:
            # Add cost metrics to the result
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
            
        return result
    
    def cleanup(self) -> None:
        """Cleanup XMODE resources."""
        # Print cost summary
        cost_summary = self.cost_tracker.get_cost_summary()
        print("\nTotal API usage summary:")
        print(f"Total cost: ${cost_summary['total_cost']:.6f}")
        print(f"Total tokens: {cost_summary['total_tokens']}")
        print(f"Total API calls: {cost_summary['total_calls']}")
        print("\nModel breakdown:")
        for model, stats in cost_summary['models'].items():
            print(f"  {model}: ${stats['cost']:.6f} ({stats['calls']} calls)")
        print("\nEndpoint breakdown:")
        for endpoint, stats in cost_summary['endpoints'].items():
            print(f"  {endpoint}: ${stats['cost']:.6f} ({stats['calls']} calls)")
