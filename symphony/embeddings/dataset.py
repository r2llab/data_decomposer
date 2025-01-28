import os
import json
import pandas as pd
from typing import List, Dict, Any, Iterator
from ..utils.text_serialization import serialize_item

class CrossModalDataset:
    def __init__(self, items: List[Dict[str, Any]]):
        """
        Initialize dataset with a list of items.
        
        Args:
            items: List of dictionaries, where each dict contains:
                  - 'data': The actual item (DataFrame or string)
                  - 'type': Type of the item ("table" or "text")
        """
        self.items = items
        
    def __len__(self) -> int:
        return len(self.items)
        
    def __getitem__(self, idx: int) -> str:
        """Get serialized string representation of item at given index."""
        item = self.items[idx]
        return serialize_item(item['data'], item['type'])
        
    def __iter__(self) -> Iterator[str]:
        """Iterate over serialized string representations of all items."""
        for item in self.items:
            yield serialize_item(item['data'], item['type'])
            
    @classmethod
    def from_directory(cls, directory: str) -> 'CrossModalDataset':
        """
        Load dataset from a directory containing tables and text files.
        
        Args:
            directory: Path to directory containing the data files
            
        Returns:
            CrossModalDataset: Dataset instance
        """
        items = []
        
        # Walk through directory
        for root, _, files in os.walk(directory):
            for file in files:
                if not file.endswith('.json'):
                    continue
                    
                filepath = os.path.join(root, file)
                
                # Load JSON file
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Handle tables directory
                if 'traindev_tables_tok' in root:
                    if isinstance(data, dict) and 'data' in data and 'header' in data:
                        # Extract column names from header
                        columns = [col[0] for col in data['header']]
                        
                        # Extract values from nested data structure
                        rows = []
                        for row in data['data']:
                            # Each row is a list of [value, links] pairs
                            # We only want the values
                            row_values = [cell[0] for cell in row]
                            rows.append(row_values)
                            
                        # Create DataFrame
                        df = pd.DataFrame(rows, columns=columns)
                        items.append({
                            'data': df,
                            'type': 'table'
                        })
                
                # Handle text directory
                elif 'traindev_request_tok' in root:
                    # Each key-value pair in the JSON is a text item
                    for text in data.values():
                        items.append({
                            'data': text,
                            'type': 'text'
                        })
                    
        print(f"Loaded {len(items)} items ({sum(1 for item in items if item['type'] == 'table')} tables, {sum(1 for item in items if item['type'] == 'text')} texts)")
        return cls(items) 