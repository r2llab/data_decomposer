import numpy as np
import pandas as pd
from typing import List, Union, Dict, Any
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a pre-trained model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                      Default is all-MiniLM-L6-v2 which is fast and good for semantic similarity.
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for input text using the sentence transformer model.
        
        Args:
            text: Single string or list of strings to embed
            
        Returns:
            np.ndarray: Embedding vectors with shape (n_samples, embedding_dim)
        """
        if isinstance(text, str):
            text = [text]
        
        # Generate embeddings using the model
        embeddings = self.model.encode(text, convert_to_numpy=True)
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