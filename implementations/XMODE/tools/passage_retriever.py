import os
from typing import List, Optional, Dict, Any, Union
import numpy as np
import faiss
import json
from pathlib import Path
import openai
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

_DESCRIPTION = (
    "passage_retriever(query: str, top_k: int = 5, context: Optional[List[str]] = None) -> Dict[str, Any]\n"
    "This tool retrieves the most relevant passages related to a search query using vector similarity search.\n"
    " - The input for this tool should be a textual `query`\n"
    " - You can specify `top_k` to control how many results to return (default is 5)\n" 
    " - You can optionally provide a list of strings as `context` to help with understanding the query\n"
    "Use this tool when the task involves searching over unstructured text documents like publications or research papers.\n"
    "The retrieved passages can be used to answer questions or provide insights about specific topics."
)

def get_embedding(text, model="text-embedding-3-small"):
    """Get embeddings for a text using OpenAI's API."""
    response = openai.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

def get_passage_retriever_tool(llm: ChatOpenAI, embeddings_dir: Path, index_dir: Path):
    """
    Creates a passage retriever tool for XMODE.
    
    Args:
        llm: The language model to use
        embeddings_dir: Directory containing embeddings and metadata
        index_dir: Directory containing the FAISS index
    
    Returns:
        A StructuredTool that can be used in the XMODE pipeline
    """
    # Load embeddings and metadata
    embedding_path = embeddings_dir / "passage_embeddings.npy"
    metadata_path = embeddings_dir / "passage_metadata.json"
    index_path = index_dir / "passage_index.faiss"
    
    # Check if files exist
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embeddings file not found at {embedding_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found at {index_path}")
    
    # Load metadata
    with open(metadata_path, "r") as f:
        passage_metadata = json.load(f)
    
    # Load FAISS index
    index = faiss.read_index(str(index_path))
    
    def passage_retriever(
        query: str,
        top_k: int = 5,
        context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve passages related to a query using vector similarity search.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            context: Optional additional context to help understand the query
            
        Returns:
            Dictionary containing search results and status
        """
        try:
            # Combine query with context if provided
            search_query = query
            if context:
                if isinstance(context, list):
                    context_str = "\n".join(context)
                    search_query = f"{query}\nContext: {context_str}"
                else:
                    search_query = f"{query}\nContext: {context}"
            
            # Get embedding for the query
            query_embedding = get_embedding(search_query)
            query_embedding_np = np.array([query_embedding], dtype=np.float32)
            
            # Search the index
            distances, indices = index.search(query_embedding_np, top_k)
            
            # Format results
            results = []
            passage_sources = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or str(idx) not in passage_metadata:
                    continue
                    
                metadata = passage_metadata[str(idx)]
                
                # Extract document ID/name from path or use the ID directly
                doc_id = metadata.get("id", f"unknown_{idx}")
                if "path" in metadata:
                    doc_path = metadata.get("path", "")
                    doc_name = os.path.basename(doc_path)
                    passage_sources.append(doc_name)
                else:
                    passage_sources.append(doc_id)
                
                results.append({
                    "passage_id": doc_id,
                    "score": float(distances[0][i]),
                    "text": metadata.get("text", ""),
                    "source": metadata.get("path", "")
                })
            
            # Return in a format compatible with XMODE's document_sources tracking
            return {
                "status": "success",
                "results": results,
                "query": query,
                "sources_used": passage_sources,
                "passages_used": passage_sources,  # Include both formats for compatibility
                "additional_kwargs": {
                    "passage_sources": passage_sources
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "query": query,
                "sources_used": [],
                "passages_used": []
            }
    
    return StructuredTool.from_function(
        name="passage_retriever",
        func=passage_retriever,
        description=_DESCRIPTION,
    )
