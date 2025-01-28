from typing import List, Dict, Any, Optional
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from dotenv import load_dotenv
import json
import pandas as pd

load_dotenv()

class Decomposer:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the decomposer with OpenAI API.
        
        Args:
            api_key: OpenAI API key. If None, loads from environment variable.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def decompose_query(self, 
                       query: str, 
                       relevant_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Decompose a complex query into sub-queries based on available items.
        
        Args:
            query: The original query
            relevant_items: List of relevant items found by discovery
            
        Returns:
            Dict containing sub-queries and execution plan
        """
        # Prepare context about available items
        items_context = self._prepare_items_context(relevant_items)
        
        # Create the prompt
        prompt = self._create_decomposition_prompt(query, items_context)
        
        # Get decomposition from GPT
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a query decomposition expert. 
                Your task is to break down complex queries into sub-queries that can be 
                answered using the available data items. Return your response in JSON format."""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        try:
            decomposition = json.loads(response.choices[0].message.content)
            return self._validate_decomposition(decomposition)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse GPT response as JSON")
            
    def _prepare_items_context(self, items: List[Dict[str, Any]]) -> str:
        """
        Prepare a context string describing the available items.
        
        Args:
            items: List of items with their metadata
            
        Returns:
            String describing the items and their types
        """
        contexts = []
        for i, item in enumerate(items, 1):
            item_type = item.get('type', 'unknown')
            content = item.get('data') if item.get('data') is not None else item.get('content')
            
            if item_type == 'table' and isinstance(content, pd.DataFrame):
                # For tables, include column names
                contexts.append(f"Item {i}: A table with columns: {', '.join(content.columns.tolist())}")
            else:
                # For text or other types, include a preview
                preview = str(content)[:100] + "..." if content else "No content available"
                contexts.append(f"Item {i}: {item_type} content: {preview}")
                
        return "\n".join(contexts)
        
    def _create_decomposition_prompt(self, query: str, items_context: str) -> str:
        """Create the prompt for GPT decomposition."""
        return f"""Given the following query and available data items, break down the query into sub-queries.

Query: {query}

Available Data Items:
{items_context}

For each sub-query, you MUST specify which item from the available items should be used to answer it (using the item number).
If you can't determine which item to use for a sub-query, use item 1 as default.

Return a JSON object with the following structure:
{{
    "requires_decomposition": boolean,  // Whether the query needs to be decomposed
    "reasoning": string,  // Explanation of why/how the query is decomposed
    "sub_queries": [  // List of sub-queries, empty if requires_decomposition is false
        {{
            "sub_query": string,  // The sub-query text
            "target_item_index": number,  // Index of the relevant item (1-based). REQUIRED. Default to 1 if unsure.
            "expected_answer_type": string  // Type of answer expected (text, number, date, etc.)
        }}
    ],
    "aggregation_strategy": string  // How to combine sub-query results, null if no aggregation needed
}}

IMPORTANT: Each sub-query MUST have a target_item_index specified. If you're unsure which item to use, default to 1."""
        
    def _validate_decomposition(self, decomposition: Dict) -> Dict:
        """Validate and clean up the decomposition response."""
        required_fields = {
            "requires_decomposition": bool,
            "reasoning": str,
            "sub_queries": list,
            "aggregation_strategy": (str, type(None))
        }
        
        for field, field_type in required_fields.items():
            if field not in decomposition:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(decomposition[field], field_type):
                raise ValueError(f"Invalid type for field {field}")
                
        if decomposition["requires_decomposition"]:
            for i, sub_query in enumerate(decomposition["sub_queries"]):
                if not all(k in sub_query for k in ["sub_query", "target_item_index", "expected_answer_type"]):
                    # If target_item_index is missing, default to 1
                    if "target_item_index" not in sub_query:
                        sub_query["target_item_index"] = 1
                    if not all(k in sub_query for k in ["sub_query", "expected_answer_type"]):
                        raise ValueError(f"Invalid sub-query structure in sub-query {i+1}")
                if not isinstance(sub_query["target_item_index"], (int, float)):
                    sub_query["target_item_index"] = 1
                    
        return decomposition 