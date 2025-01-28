import pandas as pd
import pytest
from symphony.embeddings.dataset import CrossModalDataset

def test_dataset_iteration():
    # Create sample data
    table = pd.DataFrame({
        'col1': [1, 2],
        'col2': ['a', 'b']
    })
    text = "Sample text document"
    
    items = [
        {'data': table, 'type': 'table'},
        {'data': text, 'type': 'text'}
    ]
    
    # Create dataset
    dataset = CrossModalDataset(items)
    
    # Test length
    assert len(dataset) == 2
    
    # Test iteration
    serialized_items = list(dataset)
    assert len(serialized_items) == 2
    
    # Check table serialization
    assert "1 | a || 2 | b" == serialized_items[0]
    
    # Check text serialization
    assert "Sample text document" == serialized_items[1]
    
    # Test indexing
    assert dataset[0] == serialized_items[0]
    assert dataset[1] == serialized_items[1] 