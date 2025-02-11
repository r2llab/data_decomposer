import os
import json
import time
import pandas as pd
from typing import List, Dict, Any, Iterator
from ..utils.text_serialization import serialize_item

class CrossModalDataset:
    def __init__(self, items: List[Dict[str, Any]], batch_size: int = 1):
        """
        Initialize dataset with a list of items.
        
        Args:
            items: List of dictionaries, where each dict contains:
                  - 'data': The actual item (DataFrame or string)
                  - 'type': Type of the item ("table" or "text")
            batch_size: Number of items to process at once for embeddings
        """
        self.items = items
        self.batch_size = batch_size
        self.delay = 3  # Start with 3 second delay
        self.max_delay = 60  # Maximum delay of 60 seconds
        
    def __len__(self) -> int:
        return len(self.items)
        
    def __getitem__(self, idx: int) -> str:
        """Get serialized string representation of item at given index."""
        item = self.items[idx]
        if item['type'] == 'table':
            # Convert DataFrame to string representation
            df = item['data']
            # Convert to string with a readable format
            text = f"Table with {len(df)} rows and {len(df.columns)} columns.\n"
            text += "Columns: " + ", ".join(df.columns) + "\n"
            # Add first few rows as preview
            preview_rows = min(5, len(df))
            text += df.head(preview_rows).to_string()
            return text
        else:
            return item['data']
        
    def __iter__(self) -> Iterator[str]:
        """Iterate over serialized string representations of all items in batches."""
        total_items = len(self.items)
        processed = 0
        batch = []
        current_delay = self.delay
        
        for item in self.items:
            batch.append(serialize_item(item['data'], item['type']))
            if len(batch) >= self.batch_size:
                processed += len(batch)
                print(f"Processing items {processed}/{total_items} ({(processed/total_items)*100:.1f}%)")
                
                # Keep trying to yield the batch with exponential backoff
                while True:
                    try:
                        yield from batch
                        # If successful, reset delay
                        current_delay = self.delay
                        break
                    except Exception as e:
                        if "429" in str(e):  # Rate limit error
                            print(f"Rate limit hit, waiting {current_delay} seconds...")
                            time.sleep(current_delay)
                            # Increase delay exponentially
                            current_delay = min(current_delay * 2, self.max_delay)
                        else:
                            # If it's not a rate limit error, re-raise
                            raise
                
                batch = []
                # Add a delay between batches to respect rate limits
                time.sleep(current_delay)
                
        if batch:  # Yield any remaining items
            processed += len(batch)
            print(f"Processing items {processed}/{total_items} ({(processed/total_items)*100:.1f}%)")
            
            # Same exponential backoff for the last batch
            while True:
                try:
                    yield from batch
                    break
                except Exception as e:
                    if "429" in str(e):
                        print(f"Rate limit hit, waiting {current_delay} seconds...")
                        time.sleep(current_delay)
                        current_delay = min(current_delay * 2, self.max_delay)
                    else:
                        raise
            
    @classmethod
    def from_directory(cls, directory: str, batch_size: int = 1) -> 'CrossModalDataset':
        """
        Load dataset from a directory containing tables and text files.
        
        Args:
            directory: Path to directory containing the data files
            batch_size: Number of items to process at once for embeddings
            
        Returns:
            CrossModalDataset: Dataset instance
        """
        items = []
        
        # Walk through directory
        for root, _, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                
                # Skip zip files
                if file.endswith('.zip'):
                    continue
                
                # Handle CSV files
                if file.endswith('.csv'):
                    try:
                        df = pd.read_csv(filepath)
                        items.append({
                            'data': df,
                            'type': 'table'
                        })
                    except Exception as e:
                        print(f"Error reading CSV file {filepath}: {e}")
                        continue
                
                # Handle text files
                elif file.endswith(('.txt', '.md')):
                    try:
                        with open(filepath, 'r') as f:
                            text = f.read()
                        items.append({
                            'data': text,
                            'type': 'text'
                        })
                    except Exception as e:
                        print(f"Error reading text file {filepath}: {e}")
                        continue
                
                # Handle all other files as text (except binary files)
                else:
                    try:
                        # Try to detect if file is binary
                        with open(filepath, 'rb') as f:
                            is_binary = bool(f.read(1024).translate(None, bytes([7,8,9,10,12,13,27,32,133,160]) + bytes(range(256))[14:31] + bytes(range(256))[127:]))
                            
                        if not is_binary:
                            with open(filepath, 'r') as f:
                                text = f.read()
                            items.append({
                                'data': text,
                                'type': 'text'
                            })
                    except Exception as e:
                        print(f"Error reading file {filepath}: {e}")
                        continue
                    
        print(f"Loaded {len(items)} items ({sum(1 for item in items if item['type'] == 'table')} tables, {sum(1 for item in items if item['type'] == 'text')} texts)")
        return cls(items, batch_size=batch_size) 