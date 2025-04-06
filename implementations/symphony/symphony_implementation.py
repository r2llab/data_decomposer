from pathlib import Path
from typing import Any, Dict, Optional
from core.base_implementation import BaseImplementation
from .core import Pipeline
from .utils.cost_tracker import CostTracker

class SymphonyImplementation(BaseImplementation):
    """Symphony implementation of the BaseImplementation interface."""
    
    def initialize(self) -> None:
        print("Initializing Symphony")
        """Initialize Symphony resources."""
        # Extract config values
        self.openai_api_key = self.config.get('openai_api_key')
        self.index_path = self.config.get('index_path')
        if self.index_path:
            self.index_path = Path(self.index_path)
        
        # Create cost tracker
        self.cost_tracker = CostTracker()
        
        # If data path is provided and no index path, process the data first
        data_path = self.config.get('data_path')
        if data_path and not self.index_path:
            print(f"Indexing data from {data_path}")
            embeddings_dir = Path(data_path) / 'embeddings'
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            embeddings_path = embeddings_dir / 'embeddings.npy'
            
            # Create temporary pipeline for embedding generation
            temp_pipeline = Pipeline(openai_api_key=self.openai_api_key, cost_tracker=self.cost_tracker)
            
            # Generate embeddings if they don't exist
            if not embeddings_path.exists():
                print(f"Generating embeddings for {data_path}")
                embeddings = temp_pipeline.embed_data(
                    data_dir=data_path,
                    batch_size=self.config.get('batch_size', 32),
                    output_dir=str(embeddings_dir)
                )
            
            # Create index
            index_dir = Path(data_path) / 'index'
            index_dir.mkdir(parents=True, exist_ok=True)
            print(f"Indexing data from {data_path} to {index_dir}")
            temp_pipeline.index_data(
                data_dir=data_path,
                embeddings_path=str(embeddings_path),
                output_dir=str(index_dir)
            )
            self.index_path = index_dir

        # Initialize the main pipeline with the final index path
        self.pipeline = Pipeline(
            index_path=self.index_path,
            openai_api_key=self.openai_api_key,
            cost_tracker=self.cost_tracker
        )
        
    def process_query(self, query: str) -> Any:
        """Process a query using Symphony.
        
        Args:
            query: The query string to process
            
        Returns:
            Dict containing the answer and metadata including document sources
        """
        # Reset query-specific cost tracking
        self.cost_tracker.reset_query_stats()
        
        # Process the query with the pipeline
        result = self.pipeline.run_query(query)
        
        # Ensure document sources are in the result
        if 'document_sources' not in result:
            # If pipeline didn't add document sources, add them from the pipeline's tracked sources
            result['document_sources'] = list(self.pipeline.document_sources)
            
        # Log the document sources used
        print(f"Document sources used: {result.get('document_sources', [])}")
        
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
        
        print(f"Query cost: ${cost_summary['query_cost']:.6f}")
        print(f"Total tokens: {cost_summary['query_tokens']}")
        print(f"API calls: {cost_summary['query_calls']}")
        
        return result
    
    def cleanup(self) -> None:
        """Cleanup Symphony resources."""
        # Currently no cleanup needed for Symphony
        
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
        pass 