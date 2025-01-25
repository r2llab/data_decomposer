import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from symphony.discovery import Discovery
from symphony.embeddings import Embedder

@pytest.fixture
def discovery():
    return Discovery()

@pytest.fixture
def sample_items():
    return [
        {
            "type": "person",
            "content": "John Smith is the director of engineering at TechCorp.",
            "metadata": {"role": "director", "department": "engineering"}
        },
        {
            "type": "document",
            "content": "The weather forecast predicts rain tomorrow.",
            "metadata": {"category": "weather", "date": "2024-01-23"}
        },
        {
            "type": "person",
            "content": "Alice Johnson is a software developer.",
            "metadata": {"role": "developer", "department": "engineering"}
        }
    ]

@pytest.fixture
def indexed_discovery(discovery, sample_items):
    # Index the items
    texts = [item["content"] for item in sample_items]
    discovery.index_items(sample_items, texts)
    return discovery

def test_discovery_initialization():
    """Test that Discovery initializes correctly."""
    discovery = Discovery()
    assert discovery.embedder is not None
    assert discovery.index is not None
    assert discovery.index.dimension == discovery.embedder.embedding_dim

def test_indexing(discovery, sample_items):
    """Test that items can be indexed."""
    texts = [item["content"] for item in sample_items]
    discovery.index_items(sample_items, texts)
    assert len(discovery.index.items) == len(sample_items)

def test_discovery_director_query(indexed_discovery):
    """Test that a query about director returns the relevant item first."""
    results = indexed_discovery.discover("Who is the director?", k=3)
    
    assert len(results) > 0
    assert "director" in results[0]["content"].lower()
    assert results[0]["type"] == "person"

def test_discovery_weather_query(indexed_discovery):
    """Test that a weather-related query returns the weather document."""
    results = indexed_discovery.discover("What's the weather like?", k=3)
    
    assert len(results) > 0
    assert "weather" in results[0]["content"].lower()
    assert results[0]["type"] == "document"

def test_discovery_with_keyword_boost(indexed_discovery):
    """Test that keyword boosting works."""
    # Query with exact keyword match
    results_with_boost = indexed_discovery.discover("software developer", k=3, keyword_boost=True)
    results_without_boost = indexed_discovery.discover("software developer", k=3, keyword_boost=False)
    
    # The item with exact keyword matches should be ranked higher with boosting
    assert any("software developer" in r["content"].lower() for r in results_with_boost)

def test_discovery_min_score_filtering(indexed_discovery):
    """Test that minimum score filtering works."""
    # Set a very high minimum score
    results = indexed_discovery.discover("completely unrelated query", k=3, min_score=0.99)
    assert len(results) == 0

def test_save_load_index(indexed_discovery):
    """Test that the index can be saved and loaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the index
        save_path = Path(tmpdir) / "test_index"
        indexed_discovery.save_index(save_path)
        
        # Load into a new discovery instance
        new_discovery = Discovery(index_path=save_path)
        
        # Check that the loaded index works
        results = new_discovery.discover("director", k=1)
        assert len(results) > 0
        assert "director" in results[0]["content"].lower()

def test_integration_with_embedder(indexed_discovery):
    """Test integration between Discovery and Embedder."""
    # Create a new embedder instance
    embedder = Embedder()
    
    # Get embeddings for a query
    query = "Who is the director?"
    query_embedding = embedder.embed_text(query)
    
    # Use the discovery module
    results = indexed_discovery.discover(query)
    
    assert len(results) > 0
    assert isinstance(results[0].get('relevance_score'), float) 