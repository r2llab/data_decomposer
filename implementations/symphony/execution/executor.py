from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import openai
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class Executor:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the executor with OpenAI client.
        
        Args:
            api_key: OpenAI API key. If None, loads from environment variable.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
            
        self.client = openai.OpenAI(api_key=self.api_key)
        
    def execute_query(self, query: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a query on a single item.
        
        Args:
            query: The query to execute
            item: The item to query against
            
        Returns:
            Dict containing the answer and metadata
        """
        item_type = item.get('type', 'unknown')
        if item_type == 'table':
            return self._execute_table_query(query, item)
        else:
            return self._execute_text_query(query, item)
            
    def _execute_text_query(self, query: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a query on a text item."""
        # Get content from either data or content field
        context = item.get('data') if item.get('data') is not None else item.get('content')
        if context is None:
            return {
                "answer": "No content available to answer the query.",
                "confidence": 0.0,
                "source_type": item.get('type'),
                "source": None
            }
            
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a precise question answering assistant. Answer questions based solely on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a concise answer based only on the context provided. If the context doesn't contain the information needed to answer the question, say 'The context does not provide this information.'"}
            ],
            temperature=0.3
        )
        
        return {
            "answer": response.choices[0].message.content,
            "confidence": 0.8 if "context does not provide" not in response.choices[0].message.content.lower() else 0.0,
            "source_type": item.get('type'),
            "source": context[:200] + "..." if len(context) > 200 else context
        }
        
    def _execute_table_query(self, query: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a query on a table item."""
        # Get table from either data or content field
        table = item.get('data') if item.get('data') is not None else item.get('content')
        if table is None:
            return {
                "answer": "No table data available to answer the query.",
                "confidence": 0.0,
                "source_type": "table",
                "source": None
            }
            
        # Convert table to string representation
        table_str = table.to_string()
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a precise question answering assistant. Answer questions based solely on the provided table data."},
                {"role": "user", "content": f"Table Data:\n{table_str}\n\nQuestion: {query}\n\nProvide a concise answer based only on the table data provided. If the table doesn't contain the information needed to answer the question, say 'The table does not provide this information.'"}
            ],
            temperature=0.3
        )
        
        return {
            "answer": response.choices[0].message.content,
            "confidence": 0.8 if "table does not provide" not in response.choices[0].message.content.lower() else 0.0,
            "source_type": "table",
            "source": table_str
        } 