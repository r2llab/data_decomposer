from typing import List, Dict, Any, Optional, Set
import numpy as np
from pathlib import Path
import re
from ..embeddings import Embedder
from .index import VectorIndex

class Discovery:
    def __init__(self, embedder: Optional[Embedder] = None, index_path: Optional[Path] = None):
        """
        Initialize the discovery module.
        
        Args:
            embedder: Embedder instance for encoding queries
            index_path: Path to load an existing index from
        """
        self.embedder = embedder or Embedder()
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
        
    def discover(self, 
                query: str, 
                k: int = 5, 
                min_score: float = 0.5,
                keyword_boost: bool = True) -> List[Dict[str, Any]]:
        """
        Discover relevant items for a query using both keyword matching and semantic search.
        
        Args:
            query: The search query
            k: Number of results to return
            min_score: Minimum similarity score threshold
            keyword_boost: Whether to boost scores for keyword matches
            
        Returns:
            List of discovered items
        """
        if not self.index:
            raise ValueError("No items have been indexed yet")
            
        # Get query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Get initial results from vector search
        results, scores = self.index.search(query_embedding, k=k)
        
        if keyword_boost:
            # Extract keywords from query (simple approach)
            keywords = set(re.findall(r'\w+', query.lower()))
            
            # Boost scores for items with keyword matches
            boosted_results = []
            for item, score in zip(results, scores):
                # Get text representation of item
                item_text = ' '.join(str(v) for v in item.values()).lower()
                
                # Count keyword matches
                matches = sum(1 for kw in keywords if kw in item_text)
                
                # Boost score (simple linear combination)
                boosted_score = 0.7 * score + 0.3 * (matches / len(keywords))
                
                if boosted_score >= min_score:
                    item['relevance_score'] = float(boosted_score)
                    boosted_results.append(item)
                    
            # Sort by boosted scores
            boosted_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return boosted_results[:k]
        else:
            # Filter by minimum score
            filtered_results = []
            for item, score in zip(results, scores):
                if score >= min_score:
                    item['relevance_score'] = float(score)
                    filtered_results.append(item)
            return filtered_results
        
    def save_index(self, directory: Path):
        """Save the index to disk."""
        if self.index:
            self.index.save(directory)
            
    def load_index(self, directory: Path):
        """Load the index from disk."""
        self.index = VectorIndex.load(directory) 