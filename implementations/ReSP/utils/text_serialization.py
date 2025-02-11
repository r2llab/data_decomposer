import pandas as pd
from typing import Union, Dict, Any

def serialize_item(item: Union[pd.DataFrame, str, Dict[str, Any]], item_type: str = "text") -> str:
    """
    Returns a single string that concatenates the contents of a table or text item.
    
    Args:
        item: The item to serialize - can be a pandas DataFrame, string, or dictionary
        item_type: Type of the item - either "table" or "text"
        
    Returns:
        str: Serialized string representation of the item
        
    Raises:
        ValueError: If item_type is not supported or item format doesn't match type
    """
    if item_type == "table":
        if not isinstance(item, pd.DataFrame):
            raise ValueError("For item_type='table', item must be a pandas DataFrame")
            
        # Convert each row to pipe-separated string
        row_strings = []
        for _, row in item.iterrows():
            row_str = " | ".join(str(val) for val in row)
            row_strings.append(row_str)
            
        # Join rows with double pipes
        return " || ".join(row_strings)
        
    elif item_type == "text":
        if isinstance(item, str):
            # Basic text cleaning - remove special characters, extra whitespace
            text = item.strip()
            # Could add more cleaning here if needed
            return text
        else:
            raise ValueError("For item_type='text', item must be a string")
            
    else:
        raise ValueError(f"Unsupported item_type: {item_type}") 