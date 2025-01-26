import pytest
import pandas as pd
from unittest.mock import Mock, patch
from symphony.decomposition import Decomposer

@pytest.fixture
def mock_openai():
    with patch('openai.OpenAI') as mock:
        yield mock

@pytest.fixture
def decomposer(mock_openai):
    return Decomposer(api_key="test_key")

@pytest.fixture
def sample_items():
    return [
        {
            "type": "text",
            "content": "The film 'Inception' was directed by Christopher Nolan in 2010.",
            "metadata": {"genre": "sci-fi"}
        },
        {
            "type": "table",
            "content": pd.DataFrame({
                'actor': ['Leonardo DiCaprio', 'Joseph Gordon-Levitt'],
                'role': ['Cobb', 'Arthur'],
                'screen_time': [120, 90]
            })
        }
    ]

def test_decomposition_single_source(decomposer, mock_openai, sample_items):
    """Test that simple queries don't get decomposed."""
    # Mock GPT response
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(
        content="""
        {
            "requires_decomposition": false,
            "reasoning": "This query can be answered using a single source",
            "sub_queries": [],
            "aggregation_strategy": null
        }
        """
    ))]
    mock_openai.return_value.chat.completions.create.return_value = mock_response
    
    result = decomposer.decompose_query("Who directed Inception?", sample_items)
    
    assert result["requires_decomposition"] is False
    assert not result["sub_queries"]
    assert result["aggregation_strategy"] is None

def test_decomposition_multiple_sources(decomposer, mock_openai, sample_items):
    """Test decomposition of complex queries requiring multiple sources."""
    # Mock GPT response
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(
        content="""
        {
            "requires_decomposition": true,
            "reasoning": "Need to combine director and actor information",
            "sub_queries": [
                {
                    "sub_query": "Who directed the film?",
                    "target_item_index": 1,
                    "expected_answer_type": "text"
                },
                {
                    "sub_query": "Who played the main character?",
                    "target_item_index": 2,
                    "expected_answer_type": "text"
                }
            ],
            "aggregation_strategy": "combine_director_and_actor"
        }
        """
    ))]
    mock_openai.return_value.chat.completions.create.return_value = mock_response
    
    result = decomposer.decompose_query(
        "Who directed Inception and who played the main character?",
        sample_items
    )
    
    assert result["requires_decomposition"] is True
    assert len(result["sub_queries"]) == 2
    assert result["sub_queries"][0]["target_item_index"] == 1
    assert result["sub_queries"][1]["target_item_index"] == 2
    assert result["aggregation_strategy"] == "combine_director_and_actor"

# def test_invalid_gpt_response(decomposer, mock_openai, sample_items):
#     """Test handling of invalid GPT responses."""
#     # Mock invalid JSON response
#     mock_response = Mock()
#     mock_response.choices = [Mock(message=Mock(content="Invalid JSON"))]
#     mock_openai.return_value.chat.completions.create.return_value = mock_response
    
#     with pytest.raises(ValueError, match="Failed to parse GPT response as JSON"):
#         decomposer.decompose_query("Any query", sample_items)

def test_missing_api_key():
    """Test that missing API key raises error."""
    with pytest.raises(ValueError, match="OpenAI API key must be provided"):
        Decomposer(api_key=None)

def test_items_context_preparation(decomposer, sample_items):
    """Test preparation of items context for the prompt."""
    context = decomposer._prepare_items_context(sample_items)
    
    assert "Inception" in context
    assert "Christopher Nolan" in context
    assert "columns: actor, role, screen_time" in context.lower()

# def test_validation_missing_fields(decomposer, mock_openai, sample_items):
#     """Test validation of decomposition with missing fields."""
#     # Mock response missing required fields
#     mock_response = Mock()
#     mock_response.choices = [Mock(message=Mock(
#         content="""
#         {
#             "requires_decomposition": true,
#             "sub_queries": []
#         }
#         """
#     ))]
#     mock_openai.return_value.chat.completions.create.return_value = mock_response
    
#     with pytest.raises(ValueError, match="Missing required field"):
#         decomposer.decompose_query("Any query", sample_items) 