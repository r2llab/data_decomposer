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
        if len(items) != embeddings.shape[0]:
            raise ValueError("Number of items must match number of embeddings")
            
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
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
        if k > len(self.items):
            k = len(self.items)
            
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1)
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