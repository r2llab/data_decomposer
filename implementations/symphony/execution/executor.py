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
        # Print item to file for debugging/logging
        item_type = item.get('type', 'unknown')
        if item_type == 'table':
            return self._execute_table_query(query, item)
        else:
            return self._execute_text_query(query, item)
            
    def _execute_text_query(self, query: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a query on a text item."""
        # Get content from either data or content field

        context = item.get('data') if item.get('data') is not None else item.get('content')
        # Print context to file for debugging/logging
        with open('text_query_executor.txt', 'w') as f:
            f.write(str(context))
        if context is None:
            return {
                "answer": "No content available to answer the query.",
                "confidence": 0.0,
                "source_type": item.get('type'),
                "source": None
            }
            
        response = self.client.chat.completions.create(
            model="gpt-4o",
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
            
        # Work with the DataFrame directly
        if isinstance(table, pd.DataFrame):
            # Convert the DataFrame to a JSON-serializable format for the API
            # Use pandas json serialization which handles various data types properly
            table_json = table.to_json(orient='records')
            # Print table_json to file for debugging/logging
            with open('table_query_content.txt', 'w') as f:
                f.write(table_json)
            # Create a system message that instructs the model to work with JSON data
            system_message = (
                "You are a precise question answering assistant. "
                "You will be given a table in JSON format and a question about the data. "
                "Answer the question based solely on the provided table data. "
                "If the table doesn't contain the information needed to answer the question, "
                "say 'The table does not provide this information.'"
            )
            
            # Create a user message with the table data and query
            user_message = (
                f"I have a table with {table.shape[0]} rows and {table.shape[1]} columns. "
                f"The columns are: {', '.join(table.columns)}.\n\n"
                f"Here is the table data in JSON format:\n{table_json}\n\n"
                f"Question: {query}"
            )
            
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3
            )
            
            # For source, provide a brief description
            source_description = f"Table with {table.shape[0]} rows and {table.shape[1]} columns: {', '.join(table.columns)}"
            
            return {
                "answer": response.choices[0].message.content,
                "confidence": 0.8 if "table does not provide" not in response.choices[0].message.content.lower() else 0.0,
                "source_type": "table",
                "source": source_description
            }
        else:
            # If it's not a DataFrame, handle accordingly
            return {
                "answer": "The provided table is not in a supported format.",
                "confidence": 0.0,
                "source_type": "table",
                "source": str(type(table))
            } 