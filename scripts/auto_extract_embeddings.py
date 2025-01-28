import argparse
from symphony.core.pipeline import Pipeline

def main():
    parser = argparse.ArgumentParser(description='Extract embeddings using OpenAI API')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save embeddings')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding extraction')
    
    args = parser.parse_args()
    
    # Initialize pipeline and extract embeddings
    pipeline = Pipeline(openai_api_key=args.api_key)
    pipeline.embed_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main() 