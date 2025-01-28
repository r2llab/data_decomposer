#!/usr/bin/env python3

import os
import torch
import argparse
from pathlib import Path
from torch.utils.data import random_split

from symphony.embeddings.model import SymphonyAutoEncoder
from symphony.embeddings.trainer import SymphonyTrainer
from symphony.embeddings.dataset import CrossModalDataset

def main():
    parser = argparse.ArgumentParser(description='Train the Symphony cross-modal model')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save model checkpoints')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    dataset = CrossModalDataset.from_directory(args.data_dir)
    
    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Dataset split: {train_size} train, {val_size} validation")

    # Initialize model and trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SymphonyAutoEncoder()
    trainer = SymphonyTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        device=device,
        checkpoint_dir=str(output_dir)
    )

    # Train the model
    print("Starting training...")
    trainer.train(
        num_epochs=args.epochs,
        val_frequency=1
    )

if __name__ == "__main__":
    main() 