#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from symphony.core.resp_pipeline import ReSPPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_evidence(evidence_list):
    """Format a list of evidence items for display"""
    return "\n".join(f"- {item}" for item in evidence_list)

def main():
    parser = argparse.ArgumentParser(description='Run a query through the ReSP pipeline')
    parser.add_argument('--index-dir', type=str, required=True, help='Path to the index directory')
    parser.add_argument('--api-key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--max-iterations', type=int, default=5, help='Maximum number of reasoning iterations')
    parser.add_argument('--query', type=str, help='Query to process. If not provided, will use interactive mode')
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ReSPPipeline(
        index_path=Path(args.index_dir),
        openai_api_key=args.api_key,
        max_iterations=args.max_iterations
    )

    def process_query(query):
        """Process a single query and display results"""
        print("\n" + "="*80)
        print(f"Query: {query}")
        print("="*80)

        # Run query through pipeline
        result = pipeline.run_query(query)

        # Display results
        print("\nFinal Answer:", result["answer"])
        print("\nConfidence Score:", result["confidence"])
        
        if result.get("supporting_evidence"):
            print("\nSupporting Evidence:")
            print(format_evidence(result["supporting_evidence"]))

        print("\n" + "="*80 + "\n")
        return result

    if args.query:
        # Single query mode
        process_query(args.query)
    else:
        # Interactive mode
        print("\nReSP Interactive Query Mode")
        print("Enter your questions below. Type 'exit' to quit.\n")
        
        while True:
            try:
                query = input("\nEnter your question: ").strip()
                if query.lower() in ['exit', 'quit', 'q']:
                    break
                if not query:
                    continue
                    
                process_query(query)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                continue

if __name__ == "__main__":
    main() 