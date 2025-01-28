#!/usr/bin/env python3
import os
import torch
import faiss
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

from symphony.embeddings.inference import EncoderForInference
from symphony.utils.text_serialization import serialize_item

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_index(
    data_items: List[Dict[str, Any]],
    encoder: EncoderForInference,
    output_dir: str,
    batch_size: int = 32
) -> None:
    """Build and save FAISS index from data items."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize empty lists for embeddings and metadata
    all_embeddings = []
    all_metadata = []
    
    # Process items in batches
    for i in tqdm(range(0, len(data_items), batch_size), desc="Computing embeddings"):
        batch = data_items[i:i + batch_size]
        
        # Serialize items
        serialized_batch = [
            serialize_item(item["content"], item["type"])
            for item in batch
        ]
        
        # Get embeddings for batch
        batch_embeddings = torch.cat([
            encoder.get_embedding(text)
            for text in serialized_batch
        ])
        
        # Store embeddings and metadata
        all_embeddings.append(batch_embeddings.cpu().numpy())
        all_metadata.extend([
            {
                "id": item["id"],
                "type": item["type"],
                "content": item["content"]
            }
            for item in batch
        ])
    
    # Concatenate all embeddings
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    
    # Create and train FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance
    index.add(embeddings)
    
    # Save index and metadata
    faiss.write_index(index, str(output_dir / "embeddings.faiss"))
    torch.save(all_metadata, output_dir / "metadata.pt")
    
    logger.info(f"Built index with {len(all_metadata)} items")
    logger.info(f"Index and metadata saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from data items")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing data items"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save index and metadata"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    args = parser.parse_args()
    
    # Load encoder
    encoder = EncoderForInference.from_pretrained(args.model_path)
    
    # Load data items (implement this based on your data format)
    data_items = []  # TODO: Implement data loading
    
    # Build index
    build_index(
        data_items=data_items,
        encoder=encoder,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main() 