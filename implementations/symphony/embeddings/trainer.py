import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import T5Tokenizer
from typing import Optional, Dict, List
from tqdm import tqdm
import logging

from .model import SymphonyAutoEncoder

logger = logging.getLogger(__name__)

class SymphonyTrainer:
    def __init__(
        self,
        model: SymphonyAutoEncoder,
        train_dataset,
        val_dataset=None,
        tokenizer_name: str = "t5-small",
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        max_length: int = 512,
        num_workers: int = 4,
        checkpoint_dir: str = "models/crossmodal_autoencoder",
        device: Optional[str] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.num_workers = num_workers
        self.checkpoint_dir = checkpoint_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Best validation loss for model checkpointing
        self.best_val_loss = float('inf')
    
    def collate_fn(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize and collate a batch of text sequences."""
        # Tokenize all sequences in the batch
        encoded = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create labels for reconstruction (same as input_ids)
        labels = encoded.input_ids.clone()
        
        # Replace padding token id with -100 in labels (PyTorch cross entropy convention)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "labels": labels
        }
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            loss, _ = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": total_loss / num_batches})
        
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Sample batch for reconstruction visualization
        sample_batch = None
        sample_outputs = None
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Validating")
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                loss, logits = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Store sample batch for visualization
                if sample_batch is None:
                    sample_batch = batch
                    sample_outputs = logits
                
                # Update progress bar
                progress_bar.set_postfix({"loss": total_loss / num_batches})
        
        # Get sample reconstructions
        sample_reconstructions = self._get_sample_reconstructions(
            sample_batch["input_ids"],
            sample_outputs
        )
        
        return {
            "val_loss": total_loss / num_batches,
            "sample_reconstructions": sample_reconstructions
        }
    
    def _get_sample_reconstructions(
        self,
        original_ids: torch.Tensor,
        logits: torch.Tensor,
        num_samples: int = 2
    ) -> List[Dict[str, str]]:
        """Get sample reconstructions for visualization."""
        # Get predictions
        pred_ids = torch.argmax(logits, dim=-1)
        
        # Convert to text
        samples = []
        for i in range(min(num_samples, len(original_ids))):
            original_text = self.tokenizer.decode(
                original_ids[i],
                skip_special_tokens=True
            )
            reconstructed_text = self.tokenizer.decode(
                pred_ids[i],
                skip_special_tokens=True
            )
            samples.append({
                "original": original_text,
                "reconstructed": reconstructed_text
            })
        
        return samples
    
    def save_checkpoint(self, val_loss: float, epoch: int):
        """Save model checkpoint if validation loss improved."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch
            }
            torch.save(
                checkpoint,
                os.path.join(self.checkpoint_dir, "ckpt_best.pt")
            )
            logger.info(f"Saved new best model with validation loss: {val_loss:.4f}")
    
    def train(
        self,
        num_epochs: int,
        val_frequency: int = 1,
        train_val_split: float = 0.9
    ):
        """Run training loop."""
        # Create data loaders
        if self.val_dataset is None:
            # Split training data into train/val if no validation set provided
            train_size = int(len(self.train_dataset) * train_val_split)
            val_size = len(self.train_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                self.train_dataset,
                [train_size, val_size]
            )
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        
        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Training loss: {train_loss:.4f}")
            
            # Validate
            if (epoch + 1) % val_frequency == 0:
                val_metrics = self.validate(val_loader)
                logger.info(f"Validation loss: {val_metrics['val_loss']:.4f}")
                
                # Log sample reconstructions
                logger.info("\nSample Reconstructions:")
                for sample in val_metrics["sample_reconstructions"]:
                    logger.info(f"\nOriginal: {sample['original']}")
                    logger.info(f"Reconstructed: {sample['reconstructed']}")
                
                # Save checkpoint
                self.save_checkpoint(val_metrics["val_loss"], epoch) 