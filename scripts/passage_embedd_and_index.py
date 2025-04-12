#!/usr/bin/env python3

import os
import argparse
import numpy as np
import faiss
import glob
import json
import openai
from tqdm import tqdm
from pathlib import Path
import pickle

def read_target_file(file_path):
    """Read content of a target file as a passage."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    return content

def get_embedding(text, model="text-embedding-3-small"):
    """Get embeddings for a text using OpenAI's API."""
    response = openai.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

def main():
    parser = argparse.ArgumentParser(description='Embed PubMed target passages and create FAISS index')
    parser.add_argument('--targets-dir', type=str, required=True, 
                        help='Directory containing PubMed target files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save embeddings and index')
    parser.add_argument('--api-key', type=str, required=True, 
                        help='OpenAI API key')
    parser.add_argument('--batch-size', type=int, default=100, 
                        help='Number of targets to process in batch before updating index')
    
    args = parser.parse_args()
    
    # Set up OpenAI API
    openai.api_key = args.api_key
    
    # Create output directories
    output_dir = Path(args.output_dir)
    embed_dir = output_dir / "passage-embeddings"
    index_dir = output_dir / "passage-index"
    
    os.makedirs(embed_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)
    
    # Get all target files
    target_files = glob.glob(os.path.join(args.targets_dir, "Target-*"))
    print(f"Found {len(target_files)} target files")
    
    # Store file IDs and their paths
    passage_metadata = {}
    embeddings_list = []
    
    # Process files in batches
    for i, file_path in enumerate(tqdm(target_files)):
        try:
            # Extract file ID from filename
            file_id = os.path.basename(file_path)
            
            # Read the target content
            passage_text = read_target_file(file_path)
            
            # Get embedding for the passage
            embedding = get_embedding(passage_text)
            
            # Store metadata
            passage_metadata[i] = {
                "id": file_id,
                "path": file_path,
                "text": passage_text
            }
            
            # Add to embeddings list
            embeddings_list.append(embedding)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Convert to numpy array
    embeddings = np.array(embeddings_list, dtype=np.float32)
    
    # Save embeddings
    embedding_path = embed_dir / "passage_embeddings.npy"
    metadata_path = embed_dir / "passage_metadata.json"
    
    np.save(embedding_path, embeddings)
    with open(metadata_path, "w") as f:
        json.dump(passage_metadata, f)
    
    print(f"Saved {len(embeddings)} embeddings to {embedding_path}")
    
    # Create and train FAISS index
    vector_dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    
    # Add vectors to index
    index.add(embeddings)
    
    # Save index
    index_path = index_dir / "passage_index.faiss"
    faiss.write_index(index, str(index_path))
    
    print(f"Created and saved FAISS index with {index.ntotal} vectors to {index_path}")

if __name__ == "__main__":
    main()
