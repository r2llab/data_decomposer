import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import pickle
from tqdm import tqdm

class VectorIndex:
    def __init__(self, dimension: int):
        """
        Initialize a FAISS index for vector similarity search.
        
        Args:
            dimension: Dimensionality of the vectors to be indexed
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product index (normalized vectors = cosine similarity)
        self.items: List[Dict[str, Any]] = []  # Store the actual items
        
    def add_items(self, items: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Add items and their embeddings to the index.
        
        Args:
            items: List of items to index
            embeddings: Numpy array of shape (n_items, dimension) containing the embeddings
        """
        # Safety check for empty items list
        if len(items) == 0:
            print("Warning: No items to add to the index.")
            return
            
        if len(items) != embeddings.shape[0]:
            raise ValueError("Number of items must match number of embeddings")
            
        # Convert embeddings to float32 and ensure it's contiguous in memory
        embeddings = np.ascontiguousarray(embeddings.astype('float32'))
        
        # Check for NaN or Inf values that could cause normalization issues
        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            print("Warning: Embeddings contain NaN or Inf values. Replacing with zeros.")
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Check for zero vectors that would cause normalization issues
        zero_norms = np.sum(embeddings**2, axis=1) == 0
        if zero_norms.any():
            print(f"Warning: {zero_norms.sum()} embeddings have zero norm. Adding small random noise.")
            # Add small random noise to zero vectors
            embeddings[zero_norms] = np.random.randn(zero_norms.sum(), embeddings.shape[1]).astype('float32') * 1e-5
        
        try:
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
        except Exception as e:
            print(f"Error during normalization: {e}")
            print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
            print("Attempting alternative normalization...")
            
            # Alternative normalization using numpy
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store items
        self.items.extend(items)
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Search for the k most similar items to the query.
        
        Args:
            query_embedding: Query embedding vector of shape (1, dimension)
            k: Number of results to return
            
        Returns:
            Tuple of (list of items, similarity scores)
        """
        # Handle the case when index is empty
        if len(self.items) == 0:
            print("Warning: Attempting to search an empty index")
            return [], np.array([])
            
        if k > len(self.items):
            k = len(self.items)
            
        # Ensure k is at least 1 to avoid assertion error
        k = max(1, k)
            
        # Ensure query embedding is in the right shape (1, dimension)
        if len(query_embedding.shape) == 1:
            # If 1D array, reshape to 2D
            query_embedding = query_embedding.reshape(1, -1)
        elif query_embedding.shape[0] != 1:
            # If already 2D but with wrong first dimension, take the first row
            query_embedding = query_embedding[0:1, :]

        # Ensure the embedding is float32 and contiguous
        query_embedding = np.ascontiguousarray(query_embedding.astype('float32'))
            
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Get the corresponding items
        results = [self.items[idx] for idx in indices[0]]
        
        return results, scores[0]
    
    def save(self, directory: Path):
        """Save the index and items to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(directory / "index.faiss"))
        
        # Save items
        with open(directory / "items.pkl", "wb") as f:
            pickle.dump(self.items, f)
            
    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        """Load an index from disk."""
        directory = Path(directory)
        
        # Load FAISS index
        index = faiss.read_index(str(directory / "index.faiss"))
        
        # Load items
        with open(directory / "items.pkl", "rb") as f:
            items = pickle.load(f)
            
        # Create instance and restore state
        instance = cls(index.d)
        instance.index = index
        instance.items = items
        
        return instance 