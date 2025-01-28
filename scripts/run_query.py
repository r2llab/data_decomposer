#!/usr/bin/env python3

import argparse
from pathlib import Path
from symphony.core.pipeline import Pipeline

def main():
    parser = argparse.ArgumentParser(description='Run a query through the Symphony pipeline')
    parser.add_argument('--index-dir', type=str, required=True, help='Path to the index directory')
    parser.add_argument('--api-key', type=str, required=True, help='OpenAI API key')
    args = parser.parse_args()

    # Initialize pipeline with correct parameter name
    pipeline = Pipeline(index_path=Path(args.index_dir), openai_api_key=args.api_key)

    # Define query
    query = "What motorways run along the city where 10,000 metres was ran in 26:51.11 by the male athlete Yigrem Demelash?"

    # Run query
    result = pipeline.run_query(query)

    # Print results with proper null checks
    print("\nQuery:", query)
    print("\nAnswer:", result["answer"])
    print("\nConfidence:", result["confidence"])
    
    if result.get("source_type"):
        if isinstance(result["source_type"], list):
            print("\nSource Types:", ", ".join(result["source_type"]))
        else:
            print("\nSource Type:", result["source_type"])
    
    if result.get("source"):
        if isinstance(result["source"], list):
            print("\nSources:", "\n".join(str(s) for s in result["source"]))
        else:
            print("\nSource:", result["source"])

if __name__ == "__main__":
    main() 