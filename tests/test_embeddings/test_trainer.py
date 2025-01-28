import torch
import pytest
from torch.utils.data import Dataset
from symphony.embeddings.model import SymphonyAutoEncoder
from symphony.embeddings.trainer import SymphonyTrainer

class DummyDataset(Dataset):
    """Dummy dataset for testing."""
    def __init__(self, num_samples=10):
        self.samples = [
            f"This is test sample {i} with some additional text to make it longer."
            for i in range(num_samples)
        ]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def test_trainer_initialization():
    model = SymphonyAutoEncoder()
    dataset = DummyDataset()
    trainer = SymphonyTrainer(model, dataset)
    
    assert trainer.model is model
    assert trainer.train_dataset is dataset
    assert trainer.batch_size == 8
    assert trainer.learning_rate == 1e-4
    assert trainer.max_length == 512

def test_collate_fn():
    model = SymphonyAutoEncoder()
    dataset = DummyDataset(num_samples=2)
    trainer = SymphonyTrainer(model, dataset)
    
    # Test collation of two samples
    batch = [dataset[0], dataset[1]]
    collated = trainer.collate_fn(batch)
    
    # Basic checks
    assert "input_ids" in collated
    assert "attention_mask" in collated
    assert "labels" in collated
    assert isinstance(collated["input_ids"], torch.Tensor)
    assert isinstance(collated["attention_mask"], torch.Tensor)
    assert isinstance(collated["labels"], torch.Tensor)
    assert collated["input_ids"].shape[0] == 2  # batch size
    
    # Check attention mask properties
    attention_mask = collated["attention_mask"]
    # Each sequence should have at least some valid tokens (1s in attention mask)
    assert torch.all(torch.sum(attention_mask, dim=1) > 0)
    # Attention mask should be binary (only 0s and 1s)
    assert set(attention_mask.unique().tolist()) <= {0, 1}
    # First tokens should be valid (1s) as they contain actual content
    assert torch.all(attention_mask[:, 0] == 1)

def test_training_step():
    model = SymphonyAutoEncoder()
    dataset = DummyDataset(num_samples=4)
    trainer = SymphonyTrainer(
        model,
        dataset,
        batch_size=2,
        num_workers=0  # Use 0 workers for testing
    )
    
    # Run one epoch
    train_loss = trainer.train_epoch(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=trainer.collate_fn
        )
    )
    
    assert isinstance(train_loss, float)
    assert not torch.isnan(torch.tensor(train_loss))

def test_validation_step():
    model = SymphonyAutoEncoder()
    dataset = DummyDataset(num_samples=4)
    trainer = SymphonyTrainer(
        model,
        dataset,
        batch_size=2,
        num_workers=0  # Use 0 workers for testing
    )
    
    # Run validation
    val_metrics = trainer.validate(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=trainer.collate_fn
        )
    )
    
    assert "val_loss" in val_metrics
    assert "sample_reconstructions" in val_metrics
    assert isinstance(val_metrics["val_loss"], float)
    assert len(val_metrics["sample_reconstructions"]) > 0
    assert "original" in val_metrics["sample_reconstructions"][0]
    assert "reconstructed" in val_metrics["sample_reconstructions"][0]

def test_full_training_loop():
    model = SymphonyAutoEncoder()
    dataset = DummyDataset(num_samples=4)
    trainer = SymphonyTrainer(
        model,
        dataset,
        batch_size=2,
        num_workers=0,  # Use 0 workers for testing
        checkpoint_dir="test_checkpoints"  # Temporary directory for testing
    )
    
    # Run training for 2 epochs
    trainer.train(num_epochs=2, val_frequency=1)
    
    # Check that checkpoint was saved
    assert trainer.best_val_loss != float('inf') 