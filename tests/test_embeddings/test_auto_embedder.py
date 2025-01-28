import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from symphony.embeddings.auto_embedder import AutoEmbedder

# Mock OpenAI API response
class MockEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [MagicMock(embedding=embedding) for embedding in embeddings]

@pytest.fixture
def mock_embeddings():
    # Create mock embeddings for testing
    return np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])

@pytest.fixture
def embedder():
    return AutoEmbedder(api_key="test_key", model="test-model")

def test_init():
    """Test embedder initialization"""
    embedder = AutoEmbedder(api_key="test_key")
    assert embedder.model == "text-embedding-3-small"
    assert embedder.embedding_dim == 1536

    embedder = AutoEmbedder(api_key="test_key", model="text-embedding-3-large")
    assert embedder.model == "text-embedding-3-large"
    assert embedder.embedding_dim == 1536

@patch('openai.embeddings.create')
def test_embed_text_single(mock_create, embedder, mock_embeddings):
    """Test embedding single text input"""
    mock_create.return_value = MockEmbeddingResponse([mock_embeddings[0]])
    
    result = embedder.embed_text("test text")
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3)
    np.testing.assert_array_equal(result[0], mock_embeddings[0])
    
    mock_create.assert_called_once_with(
        model=embedder.model,
        input=["test text"]
    )

@patch('openai.embeddings.create')
def test_embed_text_batch(mock_create, embedder, mock_embeddings):
    """Test embedding batch of texts"""
    mock_create.return_value = MockEmbeddingResponse(mock_embeddings)
    
    texts = ["text1", "text2", "text3"]
    result = embedder.embed_text(texts)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    np.testing.assert_array_equal(result, mock_embeddings)
    
    mock_create.assert_called_once_with(
        model=embedder.model,
        input=texts
    )

@patch('openai.embeddings.create')
def test_embed_tabular(mock_create, embedder, mock_embeddings):
    """Test embedding tabular data"""
    mock_create.return_value = MockEmbeddingResponse(mock_embeddings)
    
    df = pd.DataFrame({
        'col1': ['a', 'b', 'c'],
        'col2': [1, 2, 3]
    })
    
    result = embedder.embed_tabular(df)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    np.testing.assert_array_equal(result, mock_embeddings)
    
    expected_texts = [
        "col1: a col2: 1",
        "col1: b col2: 2",
        "col1: c col2: 3"
    ]
    mock_create.assert_called_once_with(
        model=embedder.model,
        input=expected_texts
    )

@patch('openai.embeddings.create')
def test_embed_tabular_selected_columns(mock_create, embedder, mock_embeddings):
    """Test embedding tabular data with selected columns"""
    mock_create.return_value = MockEmbeddingResponse(mock_embeddings)
    
    df = pd.DataFrame({
        'col1': ['a', 'b', 'c'],
        'col2': [1, 2, 3],
        'col3': ['x', 'y', 'z']
    })
    
    result = embedder.embed_tabular(df, columns=['col1', 'col3'])
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    
    expected_texts = [
        "col1: a col3: x",
        "col1: b col3: y",
        "col1: c col3: z"
    ]
    mock_create.assert_called_once_with(
        model=embedder.model,
        input=expected_texts
    )

def compute_cosine_similarity(v1, v2):
    """Helper function to compute cosine similarity between two vectors"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / (norm1 * norm2)

def test_compute_similarity(embedder, mock_embeddings):
    """Test computing cosine similarity between embeddings"""
    embeddings1 = mock_embeddings[:2]  # (2, 3)
    embeddings2 = mock_embeddings[1:]  # (2, 3)
    
    similarity = embedder.compute_similarity(embeddings1, embeddings2)
    
    assert isinstance(similarity, np.ndarray)
    assert similarity.shape == (2, 2)
    
    # Compute expected similarities manually using the helper function
    expected = np.array([
        [compute_cosine_similarity(embeddings1[0], embeddings2[0]),
         compute_cosine_similarity(embeddings1[0], embeddings2[1])],
        [compute_cosine_similarity(embeddings1[1], embeddings2[0]),
         compute_cosine_similarity(embeddings1[1], embeddings2[1])]
    ])
    
    np.testing.assert_array_almost_equal(similarity, expected)

@patch('openai.embeddings.create')
def test_retry_on_failure(mock_create, embedder):
    """Test retry mechanism on API failure"""
    mock_create.side_effect = [
        Exception("API Error"),
        Exception("API Error"),
        MockEmbeddingResponse([np.array([0.1, 0.2, 0.3])])
    ]
    
    result = embedder.embed_text("test text")
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3)
    assert mock_create.call_count == 3  # Should retry twice before succeeding 