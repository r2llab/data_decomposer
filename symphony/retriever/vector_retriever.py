from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
from ..embeddings import AutoEmbedder
from ..discovery import Discovery
from ..discovery.index import VectorIndex

class VectorRetriever:
    """Vector-based implementation of the ReSP Retriever component"""
    
    def __init__(self, 
                 index_path: Path,
                 api_key: Optional[str] = None,
                 top_k: int = 5,
                 min_score: float = 0.5):
        """
        Initialize the Vector Retriever.
        
        Args:
            index_path: Path to the vector index
            api_key: Optional API key for embedder
            top_k: Number of documents to retrieve
            min_score: Minimum similarity score threshold
        """
        self.embedder = AutoEmbedder(api_key=api_key)
        self.discovery = Discovery(
            embedder=self.embedder,
            index_path=index_path
        )
        self.top_k = top_k
        self.min_score = min_score
        
    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using vector similarity search.
        
        Args:
            question: The question to search for
            
        Returns:
            List of documents with their content and metadata
        """
        # Get relevant items from discovery
        relevant_items = self.discovery.discover(
            query=question,
            k=self.top_k,
            min_score=self.min_score
        )
        
        # Format results
        results = []
        for item in relevant_items:
            results.append({
                "content": item.get("data", item.get("content")),
                "metadata": {
                    "type": item.get("type"),
                    "score": item.get("relevance_score", 0.0),
                    "source": item.get("source", "unknown")
                }
            })
            
        return results 