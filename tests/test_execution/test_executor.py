import pytest
import pandas as pd
from symphony.execution.executor import Executor

@pytest.fixture
def executor():
    return Executor()

@pytest.fixture
def text_item():
    return {
        "type": "text",
        "content": "The film 'Inception' was directed by Christopher Nolan in 2010. " +
                  "It stars Leonardo DiCaprio as a professional thief who steals information " +
                  "by infiltrating the subconscious of his targets.",
        "metadata": {"genre": "sci-fi", "year": 2010}
    }

@pytest.fixture
def table_item():
    df = pd.DataFrame({
        'song': ['Bohemian Rhapsody', 'We Will Rock You', 'Another One Bites the Dust'],
        'artist': ['Queen', 'Queen', 'Queen'],
        'producer': ['Roy Thomas Baker', 'Mike Stone', 'Reinhold Mack'],
        'year': [1975, 1977, 1980]
    })
    return {
        "type": "table",
        "content": df,
        "metadata": {"source": "music_database"}
    }

def test_text_qa_director(executor, text_item):
    """Test that we can answer questions about the film director."""
    query = "Who directed Inception?"
    result = executor.execute_query(query, text_item)
    
    assert result["answer"].lower() == "christopher nolan"
    assert result["confidence"] > 0.5
    assert result["source_type"] == "text"

def test_text_qa_year(executor, text_item):
    """Test that we can answer questions about the year."""
    query = "When was the film released?"
    result = executor.execute_query(query, text_item)
    
    assert "2010" in result["answer"]
    assert result["confidence"] > 0.5

def test_text_qa_actor(executor, text_item):
    """Test that we can answer questions about the actor."""
    query = "Who stars in the film?"
    result = executor.execute_query(query, text_item)
    
    assert "leonardo dicaprio" in result["answer"].lower()
    assert result["confidence"] > 0.5

def test_table_qa_producer(executor, table_item):
    """Test that we can answer questions about song producers."""
    query = "Who produced Bohemian Rhapsody?"
    result = executor.execute_query(query, table_item)
    
    assert "roy thomas baker" in result["answer"].lower()
    assert result["confidence"] > 0.5
    assert result["source_type"] == "table"

def test_table_qa_year(executor, table_item):
    """Test that we can answer questions about song years."""
    query = "When was We Will Rock You released?"
    result = executor.execute_query(query, table_item)
    
    assert "1977" in result["answer"]
    assert result["confidence"] > 0.5

def test_table_qa_multiple(executor, table_item):
    """Test that we can answer questions about multiple songs."""
    query = "Which songs were produced by Queen?"
    result = executor.execute_query(query, table_item)
    
    # The answer should mention at least one of the songs
    assert any(song.lower() in result["answer"].lower() 
              for song in ["Bohemian Rhapsody", "We Will Rock You", "Another One Bites the Dust"])
    assert result["confidence"] > 0.5

def test_invalid_table_content(executor):
    """Test that invalid table content raises an error."""
    invalid_item = {
        "type": "table",
        "content": "Not a DataFrame"
    }
    
    with pytest.raises(ValueError):
        executor.execute_query("Any question", invalid_item) 