from typing import List, Dict, Any, Optional, Set
import numpy as np
from pathlib import Path
import re
from ..embeddings import AutoEmbedder
from .index import VectorIndex

class Discovery:
    def __init__(self, embedder: Optional[AutoEmbedder] = None, index_path: Optional[Path] = None):
        """
        Initialize the discovery module.
        
        Args:
            embedder: Embedder instance for encoding queries
            index_path: Path to load an existing index from
        """
        self.embedder = embedder or AutoEmbedder()
        self.index: Optional[VectorIndex] = None
        
        if index_path:
            self.load_index(index_path)
        else:
            self.index = VectorIndex(self.embedder.embedding_dim)
            
    def index_items(self, items: List[Dict[str, Any]], texts: List[str]):
        """
        Index a list of items with their corresponding text representations.
        
        Args:
            items: List of items to index
            texts: List of text representations for each item
        """
        # Generate embeddings
        embeddings = self.embedder.embed_text(texts)
        
        # Add to index
        self.index.add_items(items, embeddings)
        
    def discover(self, query: str, k: int = 5, min_score: float = 0.4) -> List[Dict[str, Any]]:
        """
        Discover relevant items in the index.
        
        Args:
            query: Query string
            k: Number of items to retrieve
            min_score: Minimum relevance score threshold
            
        Returns:
            List of relevant items with relevance scores
        """
        # Check if index exists and is not empty
        if not self.index or len(self.index.items) == 0:
            print("Warning: Index is empty or not initialized. Cannot perform discovery.")
            return []
            
        # Get embedding for query
        query_embedding = self.embedder.embed_text([query])
        
        # Ensure query embedding is in the correct format (float32 and contiguous)
        query_embedding = np.ascontiguousarray(query_embedding.astype('float32'))
        
        # Search index
        results, scores = self.index.search(query_embedding, k=k)

        # Filter by score and add relevance score to items
        relevant_items = []
        for i, (item, score) in enumerate(zip(results, scores)):
            if score >= min_score:
                # Add relevance score to item
                item_with_score = dict(item)
                item_with_score["relevance_score"] = float(score)
                relevant_items.append(item_with_score)
                
        # Log results
        print(f"Found {len(relevant_items)} relevant items for query: {query}")
        
        return relevant_items
        
    def save_index(self, directory: Path):
        """Save the index to disk."""
        if self.index:
            self.index.save(directory)
            
    def load_index(self, directory: Path):
        """Load the index from disk."""
        self.index = VectorIndex.load(directory) 