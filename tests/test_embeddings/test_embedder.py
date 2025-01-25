import pytest
import numpy as np
import pandas as pd
from symphony.embeddings.embedder import Embedder

@pytest.fixture
def embedder():
    return Embedder()

def test_text_embedding_single(embedder):
    """Test embedding a single text string."""
    text = "This is a test sentence."
    embedding = embedder.embed_text(text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, embedder.embedding_dim)
    assert not np.any(np.isnan(embedding))

def test_text_embedding_batch(embedder):
    """Test embedding multiple text strings."""
    texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence with different content."
    ]
    embeddings = embedder.embed_text(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(texts), embedder.embedding_dim)
    assert not np.any(np.isnan(embeddings))

def test_tabular_embedding(embedder):
    """Test embedding tabular data."""
    df = pd.DataFrame({
        'name': ['John Doe', 'Jane Smith'],
        'age': [30, 25],
        'occupation': ['Engineer', 'Designer']
    })
    
    embeddings = embedder.embed_tabular(df)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(df), embedder.embedding_dim)
    assert not np.any(np.isnan(embeddings))

def test_tabular_embedding_selected_columns(embedder):
    """Test embedding tabular data with selected columns."""
    df = pd.DataFrame({
        'name': ['John Doe', 'Jane Smith'],
        'age': [30, 25],
        'occupation': ['Engineer', 'Designer']
    })
    
    columns = ['name', 'occupation']
    embeddings = embedder.embed_tabular(df, columns=columns)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(df), embedder.embedding_dim)
    assert not np.any(np.isnan(embeddings))

def test_semantic_similarity(embedder):
    """Test that semantically similar texts have higher similarity scores."""
    texts1 = ["The cat sits on the mat."]
    texts2 = [
        "A cat is sitting on a mat.",  # Similar meaning
        "The weather is nice today."    # Different meaning
    ]
    
    embeddings1 = embedder.embed_text(texts1)
    embeddings2 = embedder.embed_text(texts2)
    
    similarities = embedder.compute_similarity(embeddings1, embeddings2)
    
    assert similarities[0, 0] > similarities[0, 1]  # Similar should have higher score
    assert similarities.shape == (len(texts1), len(texts2))
    assert np.all(similarities >= -1.0) and np.all(similarities <= 1.0)  # Cosine similarity bounds

def test_embedding_consistency(embedder):
    """Test that same input produces consistent embeddings."""
    text = "This is a test sentence."
    
    embedding1 = embedder.embed_text(text)
    embedding2 = embedder.embed_text(text)
    
    assert np.allclose(embedding1, embedding2) 