import os
import json
import time
import pandas as pd
import hashlib
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Iterator, Optional
from ..utils.text_serialization import serialize_item

class CrossModalDataset:
    def __init__(self, items: List[Dict[str, Any]], batch_size: int = 8):
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
        self.delay = 0.1  # Start with a small delay
        self.max_delay = 10  # Maximum delay of 10 seconds
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".symphony_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
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
            serialized = serialize_item(item['data'], item['type'])
            
            # Check if we have this item cached
            item_hash = hashlib.md5(serialized.encode()).hexdigest()
            cache_path = os.path.join(self.cache_dir, f"{item_hash}.pkl")
            
            if os.path.exists(cache_path):
                # If cached, yield the cached embedding
                try:
                    with open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)
                    processed += 1
                    if processed % 10 == 0 or processed == total_items:
                        print(f"Processing items {processed}/{total_items} ({(processed/total_items)*100:.1f}%) [cached]")
                    continue  # Skip this item as it's already cached
                except Exception:
                    # If there's an error loading the cache, process normally
                    pass
            
            batch.append((serialized, item_hash))
            if len(batch) >= self.batch_size:
                processed += len(batch)
                print(f"Processing items {processed}/{total_items} ({(processed/total_items)*100:.1f}%)")
                
                # Keep trying to yield the batch with exponential backoff
                while True:
                    try:
                        for serialized_item, item_hash in batch:
                            yield serialized_item
                            # We would cache the embedding here, but that's handled by the embedding code
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
                    for serialized_item, item_hash in batch:
                        yield serialized_item
                    break
                except Exception as e:
                    if "429" in str(e):
                        print(f"Rate limit hit, waiting {current_delay} seconds...")
                        time.sleep(current_delay)
                        current_delay = min(current_delay * 2, self.max_delay)
                    else:
                        raise
    
    @staticmethod
    def _process_file(filepath: str) -> Optional[Dict[str, Any]]:
        """Process a single file and return the item if successful."""
        file = os.path.basename(filepath)
        
        # Skip zip files
        if file.endswith('.zip'):
            return None
        
        # Handle CSV files
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(filepath)
                return {
                    'data': df,
                    'type': 'table'
                }
            except Exception as e:
                print(f"Error reading CSV file {filepath}: {e}")
                return None
        
        # Handle text files
        elif file.endswith(('.txt', '.md')):
            try:
                with open(filepath, 'r') as f:
                    text = f.read()
                return {
                    'data': text,
                    'type': 'text'
                }
            except Exception as e:
                print(f"Error reading text file {filepath}: {e}")
                return None
        
        # Handle all other files as text (except binary files)
        else:
            try:
                # Try to open as text first
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                    return {
                        'data': text,
                        'type': 'text'
                    }
                except UnicodeDecodeError:
                    # If it fails with UnicodeDecodeError, it might be binary
                    pass
                    
                # Try to detect if file is binary
                with open(filepath, 'rb') as f:
                    is_binary = bool(f.read(1024).translate(None, bytes([7,8,9,10,12,13,27,32,133,160]) + bytes(range(256))[14:31] + bytes(range(256))[127:]))
                    
                if not is_binary:
                    with open(filepath, 'r', errors='ignore') as f:
                        text = f.read()
                    return {
                        'data': text,
                        'type': 'text'
                    }
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")
                return None
        
        return None
            
    @classmethod
    def from_directory(cls, directory: str, batch_size: int = 8, max_workers: int = 4) -> 'CrossModalDataset':
        """
        Load dataset from a directory containing tables and text files.
        
        Args:
            directory: Path to directory containing the data files
            batch_size: Number of items to process at once for embeddings
            max_workers: Maximum number of worker threads for parallel processing
            
        Returns:
            CrossModalDataset: Dataset instance
        """
        items = []
        all_files = []
        
        # First, collect all files
        for root, _, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                all_files.append(filepath)
        
        print(f"Found {len(all_files)} files in {directory}")
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for item in executor.map(cls._process_file, all_files):
                if item is not None:
                    items.append(item)
                    
        print(f"Loaded {len(items)} items ({sum(1 for item in items if item['type'] == 'table')} tables, {sum(1 for item in items if item['type'] == 'text')} texts)")
        return cls(items, batch_size=batch_size) 