#!/usr/bin/env python3
import sqlite3
import pandas as pd
import os
import glob
import sys
from pathlib import Path

def csv_to_sqlite(csv_directory, db_file):
    """
    Convert all CSV files in the specified directory to tables in an SQLite database.
    
    Args:
        csv_directory: Directory containing CSV files
        db_file: Path to the output SQLite database file
    """
    # Create database connection
    conn = sqlite3.connect(db_file)
    
    # Get list of all CSV files in the directory
    csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))
    
    # Track progress
    total_files = len(csv_files)
    print(f"Found {total_files} CSV files to process.")
    
    # Process each CSV file
    for i, csv_file in enumerate(csv_files, 1):
        file_name = os.path.basename(csv_file)
        table_name = file_name.replace('.csv', '').replace('-', '_')
        
        print(f"[{i}/{total_files}] Processing {file_name} -> {table_name}")
        
        try:
            # Read CSV file in chunks to handle large files
            for chunk in pd.read_csv(csv_file, chunksize=10000):
                # Replace any problematic characters in column names
                chunk.columns = [col.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '') for col in chunk.columns]
                
                # Append to SQLite table (create if not exists)
                chunk.to_sql(table_name, conn, if_exists='append', index=False)
                
            print(f"✓ Successfully imported {file_name}")
        except Exception as e:
            print(f"✗ Error processing {file_name}: {str(e)}")
    
    # Close the connection
    conn.close()
    print(f"Database created at: {db_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_dir = sys.argv[1]
    else:
        # Default to the current directory's location
        script_dir = Path(__file__).parent.absolute()
        csv_dir = os.path.join(script_dir, "drugbank-tables")
    
    # Output DB file in the same directory as the CSV files
    output_db = os.path.join(os.path.dirname(csv_dir), "drugbank.db")
    
    print(f"Converting CSVs from: {csv_dir}")
    print(f"Output database: {output_db}")
    
    csv_to_sqlite(csv_dir, output_db) 