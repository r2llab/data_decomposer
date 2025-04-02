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
            
            # Create a more compact representation for tables
            result = f"TABLE{chunk_info} with {num_rows} rows and {num_cols} columns:\n\n"
            result += "COLUMNS: " + ", ".join(df.columns) + "\n\n"
            
            # Use a more compact representation for all tables
            # For tables with many columns, use a more compact format
            if num_cols > 10:
                # For wide tables, use a more compact representation
                # Convert to records format (list of dicts) for more compact representation
                records = df.to_dict(orient='records')
                # Limit to first 50 rows if not a chunk (chunks are already small)
                if not is_chunk and num_rows > 50:
                    records = records[:50]
                    result += f"Showing first 50 of {num_rows} rows:\n\n"
                
                # Format each record as a compact string
                for i, record in enumerate(records):
                    result += f"Row {i+1}:\n"
                    for col, val in record.items():
                        # Truncate very long values
                        val_str = str(val)
                        if len(val_str) > 100:
                            val_str = val_str[:97] + "..."
                        result += f"  {col}: {val_str}\n"
                    result += "\n"
            else:
                # For tables with fewer columns, use standard to_string but with compact options
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 
                                      'display.width', 120, 'display.max_colwidth', 30):
                    # Limit to first 50 rows if not a chunk
                    if not is_chunk and num_rows > 50:
                        result += f"Showing first 50 of {num_rows} rows:\n\n"
                        result += df.head(50).to_string(index=False)
                    else:
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
                
                # Create a more compact representation for tables
                result = f"TABLE{chunk_info} with {num_rows} rows and {num_cols} columns:\n\n"
                result += "COLUMNS: " + ", ".join(df.columns) + "\n\n"
                
                # Use a more compact representation for all tables
                # For tables with many columns, use a more compact format
                if num_cols > 10:
                    # For wide tables, use a more compact representation
                    # Convert to records format (list of dicts) for more compact representation
                    records = df.to_dict(orient='records')
                    # Limit to first 50 rows if not a chunk
                    if not item.get('is_chunk') and num_rows > 50:
                        records = records[:50]
                        result += f"Showing first 50 of {num_rows} rows:\n\n"
                    
                    # Format each record as a compact string
                    for i, record in enumerate(records):
                        result += f"Row {i+1}:\n"
                        for col, val in record.items():
                            # Truncate very long values
                            val_str = str(val)
                            if len(val_str) > 100:
                                val_str = val_str[:97] + "..."
                            result += f"  {col}: {val_str}\n"
                        result += "\n"
                else:
                    # For tables with fewer columns, use standard to_string but with compact options
                    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 
                                          'display.width', 120, 'display.max_colwidth', 30):
                        # Limit to first 50 rows if not a chunk
                        if not item.get('is_chunk') and num_rows > 50:
                            result += f"Showing first 50 of {num_rows} rows:\n\n"
                            result += df.head(50).to_string(index=False)
                        else:
                            result += df.to_string(index=False)
                
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