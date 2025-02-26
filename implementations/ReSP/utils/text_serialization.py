import pandas as pd
from typing import Any, Dict, Union

def serialize_item(item: Any, item_type: str) -> str:
    """
    Convert an item to a string representation suitable for embedding.
    
    Args:
        item: The item to serialize (can be DataFrame, text, etc.)
        item_type: Type of the item ("table" or "text")
        
    Returns:
        String representation of the item
    """
    if item_type == 'table':
        if isinstance(item, pd.DataFrame):
            df = item
            
            # Get metadata if this is a chunked table
            is_chunk = False
            chunk_info = ""
            if isinstance(item, Dict) and item.get('is_chunk'):
                is_chunk = True
                chunk_info = f" (Chunk {item.get('chunk_id')} of table {item.get('original_table')})"
            
            # Create a text representation of the table
            num_rows = len(df)
            num_cols = len(df.columns)
            
            result = f"TABLE{chunk_info} with {num_rows} rows and {num_cols} columns:\n\n"
            result += "COLUMNS: " + ", ".join(df.columns) + "\n\n"
            
            # For very large tables, just include a sample
            max_rows_to_show = 100 if not is_chunk else num_rows
            if num_rows > max_rows_to_show:
                # Show header, first few rows, and last few rows
                rows_each_end = max_rows_to_show // 2
                header_rows = df.head(rows_each_end).to_string(index=False)
                footer_rows = df.tail(rows_each_end).to_string(index=False)
                result += header_rows + "\n...\n" + footer_rows
            else:
                # Show the entire table
                result += df.to_string(index=False)
            
            return result
        else:
            # Handle case where item might already be a dict containing a DataFrame
            if isinstance(item, Dict) and 'data' in item and isinstance(item['data'], pd.DataFrame):
                df = item['data']
                
                # Add chunk info if available
                chunk_info = ""
                if item.get('is_chunk'):
                    chunk_info = f" (Chunk {item.get('chunk_id')} of table {item.get('original_table')})"
                
                num_rows = len(df)
                num_cols = len(df.columns)
                
                result = f"TABLE{chunk_info} with {num_rows} rows and {num_cols} columns:\n\n"
                result += "COLUMNS: " + ", ".join(df.columns) + "\n\n"
                
                # For very large tables, just include a sample
                max_rows_to_show = min(num_rows, 100)
                result += df.head(max_rows_to_show).to_string(index=False)
                
                if num_rows > max_rows_to_show:
                    result += f"\n\n[Table truncated, showing {max_rows_to_show} of {num_rows} rows]"
                
                # Add source information
                if 'source' in item:
                    result += f"\n\nSOURCE: {item['source']}"
                
                return result
            return str(item)
    else:
        # Default case for text
        if isinstance(item, Dict) and 'data' in item and isinstance(item['data'], str):
            text = item['data']
            # Add source information if available
            if 'source' in item:
                return f"{text}\n\nSOURCE: {item['source']}"
            return text
        return str(item) 