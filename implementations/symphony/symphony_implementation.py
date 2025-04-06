from pathlib import Path
from typing import Any, Dict, Optional
from core.base_implementation import BaseImplementation
from .core import Pipeline
from .utils.cost_tracker import CostTracker
import numpy as np
from difflib import SequenceMatcher
import pandas as pd

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
        
        # Source relevance tracking
        self.ground_truth_answer = None
        self.source_relevance_scores = []
        
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
        
        # Hook into the pipeline to track source relevance
        self._patch_pipeline_for_source_relevance()
        
    def _patch_pipeline_for_source_relevance(self):
        """Patch pipeline methods to track source relevance to ground truth."""
        original_discover = self.pipeline._discover
        original_execute = self.pipeline._execute
        
        def wrapped_discover(query):
            """Wrapper for discover that tracks source relevance."""
            relevant_items = original_discover(query)
            
            # Calculate relevance scores if ground truth is available
            if self.ground_truth_answer and relevant_items:
                for item in relevant_items:
                    content = item.get('data') if item.get('data') is not None else item.get('content')
                    if content is not None:
                        relevance = self._calculate_text_similarity(content, self.ground_truth_answer)
                        self.source_relevance_scores.append(relevance)
                        print(f"Source relevance score: {relevance:.4f}")
            
            return relevant_items
        
        def wrapped_execute(query, item):
            """Wrapper for execute that tracks source relevance."""
            # Calculate relevance score if ground truth is available
            if self.ground_truth_answer:
                content = item.get('data') if item.get('data') is not None else item.get('content')
                if content is not None:
                    relevance = self._calculate_text_similarity(content, self.ground_truth_answer)
                    self.source_relevance_scores.append(relevance)
                    print(f"Source relevance score: {relevance:.4f}")
            
            return original_execute(query, item)
        
        # Replace the methods
        self.pipeline._discover = wrapped_discover
        self.pipeline._execute = wrapped_execute
    
    def _calculate_text_similarity(self, content: Any, reference: str) -> float:
        """
        Calculate the similarity between content and reference text.
        Handles different content types including DataFrames.
        
        Args:
            content: The content to compare (could be text, DataFrame, etc.)
            reference: The reference text to compare against
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Handle pandas DataFrame
        if isinstance(content, pd.DataFrame):
            try:
                # Convert DataFrame to string representation
                # Limit to first 10 rows for performance
                df_sample = content.head(10)
                text = df_sample.to_string()
                return SequenceMatcher(None, text, reference).ratio()
            except Exception as e:
                print(f"Error processing DataFrame for similarity: {e}")
                return 0.0
        # Handle string content
        elif isinstance(content, str):
            return SequenceMatcher(None, content, reference).ratio()
        # Handle other types by converting to string
        else:
            try:
                text = str(content)
                return SequenceMatcher(None, text, reference).ratio()
            except Exception as e:
                print(f"Error converting content to string for similarity: {e}")
                return 0.0
        
    def process_query(self, query: str, ground_truth_answer: Optional[str] = None) -> Any:
        """Process a query using Symphony.
        
        Args:
            query: The query string to process
            ground_truth_answer: Optional ground truth answer for relevance scoring
            
        Returns:
            Dict containing the answer and metadata including document sources
        """
        # Reset query-specific cost tracking and source relevance tracking
        self.cost_tracker.reset_query_stats()
        self.ground_truth_answer = ground_truth_answer
        self.source_relevance_scores = []
        
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
        
        # Add source relevance score if ground truth was provided
        if self.ground_truth_answer and self.source_relevance_scores:
            # Calculate average relevance score
            avg_relevance = sum(self.source_relevance_scores) / len(self.source_relevance_scores)
            # Calculate max relevance score
            max_relevance = max(self.source_relevance_scores) if self.source_relevance_scores else 0.0
            
            result['source_relevance_score'] = {
                'average': float(avg_relevance),
                'maximum': float(max_relevance),
                'scores': [float(score) for score in self.source_relevance_scores]
            }
            print(f"Source relevance: avg={avg_relevance:.4f}, max={max_relevance:.4f}")
        
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