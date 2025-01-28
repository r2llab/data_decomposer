#!/usr/bin/env python3

import argparse
from symphony.core.pipeline import Pipeline

def main():
    parser = argparse.ArgumentParser(description='Build FAISS index from Symphony embeddings')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing original data')
    parser.add_argument('--embeddings', type=str, required=True, help='Path to embeddings.npy file')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save index')
    parser.add_argument('--api-key', type=str, required=True, help='OpenAI API key')
    
    args = parser.parse_args()
    
    # Initialize pipeline and build index
    pipeline = Pipeline(openai_api_key=args.api_key)
    pipeline.index_data(
        data_dir=args.data_dir,
        embeddings_path=args.embeddings,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 