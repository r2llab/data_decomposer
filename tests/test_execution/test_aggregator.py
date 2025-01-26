import pytest
from unittest.mock import Mock, patch
from symphony.execution.aggregator import Aggregator

@pytest.fixture
def mock_openai():
    with patch('openai.OpenAI') as mock:
        yield mock

@pytest.fixture
def aggregator(mock_openai):
    return Aggregator(api_key="test_key")

@pytest.fixture
def sample_sub_queries():
    return [
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
    ]

@pytest.fixture
def sample_sub_results():
    return [
        {
            "answer": "Christopher Nolan",
            "confidence": 0.9,
            "source_type": "text"
        },
        {
            "answer": "Leonardo DiCaprio",
            "confidence": 0.85,
            "source_type": "table"
        }
    ]

def test_aggregation(aggregator, mock_openai, sample_sub_queries, sample_sub_results):
    """Test basic aggregation of multiple results."""
    # Mock GPT response
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(
        content="The film was directed by Christopher Nolan and stars Leonardo DiCaprio."
    ))]
    mock_openai.return_value.chat.completions.create.return_value = mock_response
    
    result = aggregator.aggregate_results(
        original_query="Who directed the film and who was the main actor?",
        sub_queries=sample_sub_queries,
        sub_results=sample_sub_results,
        aggregation_strategy="combine_director_and_actor"
    )
    
    assert "Christopher Nolan" in result["answer"]
    assert "Leonardo DiCaprio" in result["answer"]
    assert result["confidence"] == pytest.approx(0.875)  # Average of 0.9 and 0.85
    assert len(result["sub_results"]) == 2
    assert result["aggregation_strategy"] == "combine_director_and_actor"

def test_missing_api_key():
    """Test that missing API key raises error."""
    with pytest.raises(ValueError, match="OpenAI API key must be provided"):
        Aggregator(api_key=None)

def test_context_creation(aggregator, sample_sub_queries, sample_sub_results):
    """Test creation of aggregation context."""
    context = aggregator._create_aggregation_context(
        query="Who directed the film and who was the main actor?",
        sub_queries=sample_sub_queries,
        sub_results=sample_sub_results,
        strategy="combine_director_and_actor"
    )
    
    assert "Original Query:" in context
    assert "Aggregation Strategy: combine_director_and_actor" in context
    assert "Christopher Nolan" in context
    assert "Leonardo DiCaprio" in context
    assert "0.9" in context  # Confidence score
    assert "0.85" in context  # Confidence score

@patch('tenacity.wait_exponential', return_value=lambda x: 0)
def test_retry_on_error(mock_wait, aggregator, mock_openai, sample_sub_queries, sample_sub_results):
    """Test that aggregation retries on failure."""
    # Mock OpenAI to fail twice then succeed
    mock_openai.return_value.chat.completions.create.side_effect = [
        Exception("API Error"),
        Exception("API Error"),
        Mock(choices=[Mock(message=Mock(content="Final answer"))])
    ]
    
    result = aggregator.aggregate_results(
        original_query="test query",
        sub_queries=sample_sub_queries,
        sub_results=sample_sub_results,
        aggregation_strategy="test"
    )
    
    assert result["answer"] == "Final answer"
    assert mock_openai.return_value.chat.completions.create.call_count == 3

# def test_empty_results(aggregator, mock_openai):
#     """Test handling of empty results."""
#     with pytest.raises(ZeroDivisionError):  # Or change to handle this case gracefully
#         aggregator.aggregate_results(
#             original_query="test",
#             sub_queries=[],
#             sub_results=[],
#             aggregation_strategy="test"
#         ) 