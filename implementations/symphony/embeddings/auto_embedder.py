import numpy as np
import pandas as pd
import openai
from typing import List, Union, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

class AutoEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize the embedder with OpenAI API.
        
        Args:
            api_key: OpenAI API key
            model: Name of the OpenAI embedding model to use.
                  Default is text-embedding-3-small which offers good balance of performance and cost.
        """
        openai.api_key = api_key
        self.model = model
        # text-embedding-3-small has 1536 dimensions
        self.embedding_dim = 1536 if "text-embedding-3" in model else 1536

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for input text using OpenAI's embedding model.
        
        Args:
            text: Single string or list of strings to embed
            
        Returns:
            np.ndarray: Embedding vectors with shape (n_samples, embedding_dim)
        """
        if isinstance(text, str):
            text = [text]
        
        # Generate embeddings using OpenAI API
        response = openai.embeddings.create(
            model=self.model,
            input=text
        )
        
        # Extract embeddings from response
        embeddings = np.array([item.embedding for item in response.data])
        return embeddings

    def embed_tabular(self, df: pd.DataFrame, columns: List[str] = None) -> np.ndarray:
        """
        Generate embeddings for tabular data by converting rows to text.
        
        Args:
            df: Pandas DataFrame to embed
            columns: List of column names to include. If None, uses all columns.
            
        Returns:
            np.ndarray: Embedding vectors with shape (n_rows, embedding_dim)
        """
        if columns is None:
            columns = df.columns.tolist()
            
        # Convert each row to a text representation
        text_rows = []
        for _, row in df.iterrows():
            row_text = " ".join(f"{col}: {row[col]}" for col in columns)
            text_rows.append(row_text)
            
        # Generate embeddings for the text representations
        return self.embed_text(text_rows)

    def compute_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings with shape (n1, embedding_dim)
            embeddings2: Second set of embeddings with shape (n2, embedding_dim)
            
        Returns:
            np.ndarray: Similarity matrix with shape (n1, n2)
        """
        # Normalize the embeddings
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        embeddings1_norm = embeddings1 / norm1
        embeddings2_norm = embeddings2 / norm2
        
        # Compute cosine similarity
        return np.dot(embeddings1_norm, embeddings2_norm.T) 