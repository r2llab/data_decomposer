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
            
    @staticmethod
    def _chunk_table(df, source_id, max_rows=500):
        """
        Split large tables into manageable chunks while preserving headers.
        
        Args:
            df: pandas DataFrame to chunk
            source_id: Source identifier for the table
            max_rows: Maximum number of rows per chunk
            
        Returns:
            List of dictionaries containing chunked tables with source info
        """
        # For small tables (under the threshold), don't chunk at all
        if len(df) <= max_rows:
            # Table is small enough, no need to chunk
            return [{
                'data': df,
                'type': 'table',
                'source': source_id,
                'id': source_id,
                'is_chunk': False,
                'chunk_id': None
            }]
        
        # For larger tables, use adaptive chunking strategy
        total_rows = len(df)
        
        # # Use smaller chunks for drugbank tables
        # is_drugbank = 'drugbank' in source_id.lower()
        
        # Determine chunk size based on table size and type
        # if is_zdrugbank:
            # For drugbank tables, use much smaller chunks
        # if total_rows > 10000:
        #     adaptive_chunk_size = min(100, total_rows // 30)  # Very small chunks for very large drugbank tables
        # elif total_rows > 5000:
        #     adaptive_chunk_size = min(150, total_rows // 25)  # Small chunks for large drugbank tables
        # else:
        #     adaptive_chunk_size = min(200, total_rows // 20)  # Smaller chunks for medium drugbank tables
        adaptive_chunk_size = 100
        # else:
        #     # For non-drugbank tables, also use smaller chunks
        #     if total_rows > 10000:
        #         adaptive_chunk_size = min(200, total_rows // 20)
        #     elif total_rows > 5000:
        #         adaptive_chunk_size = min(300, total_rows // 15)
        #     else:
        #         adaptive_chunk_size = min(400, total_rows // 10)
                
        # Table needs to be chunked
        chunks = []
        total_chunks = (total_rows + adaptive_chunk_size - 1) // adaptive_chunk_size  # Ceiling division
        
        # Limit the number of chunks per table to avoid explosion of items
        # For drugbank tables, allow more chunks to ensure better granularity
        # max_chunks = 50 if is_drugbank else 30
        
        # if total_chunks > max_chunks:
        #     # For extremely large tables, limit to max_chunks
        #     adaptive_chunk_size = (total_rows + max_chunks - 1) // max_chunks  # Ensure at most max_chunks chunks
        #     total_chunks = (total_rows + adaptive_chunk_size - 1) // adaptive_chunk_size
            
        # For very wide tables, also consider column chunking
        max_cols = 20  # Maximum number of columns per chunk
        col_chunks = []
        
        # If table has many columns, split by columns too
        if len(df.columns) > max_cols:
            # First split by rows
            for i in range(0, total_rows, adaptive_chunk_size):
                row_chunk = df.iloc[i:i+adaptive_chunk_size].copy()
                
                # Then split each row chunk by columns
                for j in range(0, len(df.columns), max_cols):
                    col_subset = row_chunk.iloc[:, j:j+max_cols].copy()
                    
                    # Create a unique ID for this row+column chunk
                    row_chunk_id = f"{i//adaptive_chunk_size + 1}_of_{total_chunks}"
                    col_chunk_id = f"{j//max_cols + 1}_of_{(len(df.columns) + max_cols - 1) // max_cols}"
                    chunk_id = f"{row_chunk_id}_cols_{col_chunk_id}"
                    chunk_source = f"{source_id}:chunk_{chunk_id}"
                    
                    col_chunks.append({
                        'data': col_subset,
                        'type': 'table',
                        'source': source_id,  # Original source remains unchanged
                        'id': chunk_source,   # Unique ID for the chunk
                        'is_chunk': True,
                        'chunk_id': chunk_id,
                        'original_table': source_id
                    })
            
            if col_chunks:
                print(f"Split wide table {source_id} into {len(col_chunks)} row+column chunks")
                return col_chunks
        
        # If not splitting by columns, just split by rows
        for i in range(0, total_rows, adaptive_chunk_size):
            chunk_df = df.iloc[i:i+adaptive_chunk_size].copy()
            chunk_id = f"{i//adaptive_chunk_size + 1}_of_{total_chunks}"
            chunk_source = f"{source_id}:chunk_{chunk_id}"
            
            chunks.append({
                'data': chunk_df,
                'type': 'table',
                'source': source_id,  # Original source remains unchanged
                'id': chunk_source,   # Unique ID for the chunk
                'is_chunk': True,
                'chunk_id': chunk_id,
                'original_table': source_id
            })
        
        print(f"Split table {source_id} into {len(chunks)} chunks (max {adaptive_chunk_size} rows per chunk)")
        return chunks
        
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
        
        # Check if directory exists
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist")
            return cls(items, batch_size=batch_size)
        
        # # List of high-value tables to prioritize (these contain the most valuable information)
        # high_value_tables = [
        #     'drug_dosages', 'drug', 'targets', 'drug_pharmacology', 
        #     'drug_classifications', 'drug_categories', 'drug_international_brands',
        #     'targets_actions', 'transporters_actions', 'enzymes_actions', 'carriers_actions',
        #     'drug_food_interactions', 'drug_drug_interactions'
        # ]
        
        # # Tables to skip entirely (these typically have less semantic value for most queries)
        # low_value_tables = [
        #     'drug_pdb_entries', 'drug_external_identifiers', 'drug_external_links',
        #     'drug_attachments', 'drug_calculated_properties', 'drug_patents',
        #     'drug_prices', 'drug_atc_codes', 'drug_ahfs_codes'
        # ]
        
        # # Track statistics
        # skipped_tables = 0
        # included_tables = 0
        
        # Walk through directory
        for root, _, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                # Get relative path as source identifier
                rel_path = os.path.relpath(filepath, directory)
                # Use file basename as ID
                file_id = os.path.basename(filepath)
                
                # Skip hidden files and zip files
                if file.startswith('.') or file.endswith(('.zip', '.gz', '.tar')):
                    continue
                
                # Special case for Target files without extensions (always include these)
                if file.startswith('Target-'):
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                            text = f.read()
                        items.append({
                            'data': text,
                            'type': 'text',
                            'source': rel_path,
                            'id': file_id
                        })
                        continue  # Skip further processing for this file
                    except Exception as e:
                        print(f"Error reading Target file {filepath}: {e}")
                        continue
                
                # Check if this is a drugbank table (in drugbank-tables directory)
                # is_drugbank_table = 'drugbank-tables' in filepath or 'drugbank' in filepath
                
                # Handle table files (CSV, TSV, Excel)
                if file.lower().endswith(('.csv', '.tsv', '.xlsx', '.xls')):
                    # Apply filtering for drugbank tables
                    # if is_drugbank_table:
                    #     # Extract table name without prefix for checking
                    #     table_name = file_id.replace('.csv', '').replace('.tsv', '').replace('.xlsx', '').replace('.xls', '')
                        
                    #     # # Skip low-value tables
                    #     # if any(low_value in table_name for low_value in low_value_tables):
                    #     #     skipped_tables += 1
                    #     #     print(f"Skipping low-value table: {file_id}")
                    #     #     continue
                        
                    #     # # Prioritize high-value tables
                    #     # is_high_value = any(high_value in table_name for high_value in high_value_tables)
                    
                    # Try to read the table
                    try:
                        if file.lower().endswith('.csv'):
                            df = pd.read_csv(filepath, low_memory=False)
                        elif file.lower().endswith('.tsv'):
                            df = pd.read_csv(filepath, sep='\t', low_memory=False)
                        elif file.lower().endswith(('.xlsx', '.xls')):
                            df = pd.read_excel(filepath)
                        
                        # Skip empty tables
                        if df.empty:
                            print(f"Skipping empty table: {file_id}")
                            continue
                        
                        # For drugbank tables, apply chunking
                        # if is_drugbank_table:
                        # For high-value tables, use smaller chunks to maintain detail
                        max_rows = 100  # Reduced from 200/400 to 100/150
                        table_chunks = cls._chunk_table(df, file_id, max_rows=max_rows)
                        items.extend(table_chunks)
                        # included_tables += 1
                        # else:
                        #     # Non-drugbank tables are also chunked with reasonable sizes
                        #     max_rows = 200  # Reduced from default to ensure chunks fit in context window
                        #     table_chunks = cls._chunk_table(df, file_id, max_rows=max_rows)
                        #     items.extend(table_chunks)
                        #     included_tables += 1
                    except Exception as e:
                        print(f"Error reading table file {filepath}: {e}")
                        continue
                
                # Handle text files (expand the list of text extensions)
                elif file.lower().endswith(('.txt', '.md', '.json', '.xml', '.html', '.htm', '.py', '.js', '.java', '.c', '.cpp', '.h', '.cs', '.php')):
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                            text = f.read()
                        items.append({
                            'data': text,
                            'type': 'text',
                            'source': rel_path,
                            'id': file_id
                        })
                    except Exception as e:
                        print(f"Error reading text file {filepath}: {e}")
                        continue
                
                # Handle all other files as potential text (except binary files)
                else:
                    try:
                        # # Try to detect if file is binary
                        # with open(filepath, 'rb') as f:
                        #     chunk = f.read(1024)
                        #     is_binary = False
                        #     # Simple binary detection - if NUL bytes are present, likely binary
                        #     if b'\x00' in chunk:
                        #         is_binary = True
                        #     # Fallback check using translate
                        #     if not is_binary:
                        #         is_binary = bool(chunk.translate(None, bytes([7,8,9,10,12,13,27,32,133,160]) + 
                        #                                         bytes(range(256))[14:31] + bytes(range(256))[127:]))
                            
                        # if not is_binary:
                        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                            text = f.read()
                        items.append({
                            'data': text,
                            'type': 'text',
                            'source': rel_path,
                            'id': file_id
                        })
                    except Exception as e:
                        print(f"Error reading file {filepath}: {e}")
                        continue
        
        print(f"Loaded {len(items)} items ({sum(1 for item in items if item['type'] == 'table')} tables, {sum(1 for item in items if item['type'] == 'text')} texts)")
        # print(f"Tables included: {included_tables}, Tables skipped: {skipped_tables}")
        return cls(items, batch_size=batch_size) 