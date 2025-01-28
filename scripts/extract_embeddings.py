#!/usr/bin/env python3

import os
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

from symphony.embeddings.inference import EncoderForInference
from symphony.embeddings.dataset import CrossModalDataset

def main():
    parser = argparse.ArgumentParser(description='Extract embeddings using trained Symphony model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing data to embed')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save embeddings')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder
    print("Loading encoder...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = EncoderForInference.from_pretrained(
        checkpoint_path=args.checkpoint,
        device=device
    )
    encoder.eval()

    # Load dataset
    print("Loading dataset...")
    dataset = CrossModalDataset.from_directory(args.data_dir)
    
    # Extract embeddings in batches
    print("Extracting embeddings...")
    all_embeddings = []
    
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch = [dataset[j] for j in range(i, min(i + args.batch_size, len(dataset)))]
        
        with torch.no_grad():
            embeddings = torch.stack([
                encoder.get_embedding(item) for item in batch
            ])
            all_embeddings.append(embeddings)
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Save embeddings
    print(f"Saving {len(all_embeddings)} embeddings...")
    torch.save(all_embeddings, output_dir / "embeddings.pt")
    
    print("Done!")

if __name__ == "__main__":
    main() 