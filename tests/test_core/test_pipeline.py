import pytest
from symphony.core.pipeline import Pipeline

def test_pipeline_basic():
    """Test that the pipeline runs end-to-end without errors."""
    pipeline = Pipeline()
    query = "What is the capital of France?"
    
    # Test that the pipeline returns a non-empty string
    result = pipeline.run_query(query)
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Test with empty query
    result = pipeline.run_query("")
    assert isinstance(result, str)
    
    # Test individual components
    items = pipeline._discover(query)
    assert isinstance(items, list)
    assert len(items) > 0
    
    sub_queries = pipeline._decompose(query, items)
    assert isinstance(sub_queries, list)
    assert len(sub_queries) > 0
    
    answer = pipeline._execute(sub_queries[0])
    assert isinstance(answer, str)
    assert len(answer) > 0 