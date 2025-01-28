import torch
import pytest
import tempfile
from pathlib import Path

from symphony.embeddings.model import SymphonyAutoEncoder
from symphony.embeddings.inference import EncoderForInference

def create_dummy_checkpoint():
    """Create a dummy checkpoint for testing."""
    model = SymphonyAutoEncoder()
    
    # Create a temporary checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None,
            "val_loss": 0.0,
            "epoch": 0
        }
        torch.save(checkpoint, f.name)
        return f.name

def test_encoder_from_pretrained():
    # Create dummy checkpoint
    checkpoint_path = create_dummy_checkpoint()
    
    try:
        # Load encoder
        encoder = EncoderForInference.from_pretrained(checkpoint_path)
        
        assert isinstance(encoder, EncoderForInference)
        assert hasattr(encoder, "encoder")
        assert hasattr(encoder, "projection")
        assert hasattr(encoder, "tokenizer")
    finally:
        # Cleanup
        Path(checkpoint_path).unlink()

def test_get_embedding():
    # Create dummy checkpoint
    checkpoint_path = create_dummy_checkpoint()
    
    try:
        # Load encoder
        encoder = EncoderForInference.from_pretrained(checkpoint_path)
        
        # Test single string embedding
        test_string = "This is a test string for embedding."
        embedding = encoder.get_embedding(test_string)
        
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (1, 512)  # batch_size=1, embedding_dim=512
        assert not torch.isnan(embedding).any()
    finally:
        # Cleanup
        Path(checkpoint_path).unlink()

def test_encoder_save_load():
    # Create dummy checkpoint and load encoder
    checkpoint_path = create_dummy_checkpoint()
    encoder = EncoderForInference.from_pretrained(checkpoint_path)
    
    try:
        # Create temporary directory for saving
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "encoder.pt"
            
            # Save encoder
            encoder.save_pretrained(save_path)
            
            # Check saved files exist
            assert save_path.exists()
            assert (Path(tmp_dir) / "tokenizer").exists()
            
            # Test embedding before saving
            test_string = "Test string for consistency check."
            embedding_before = encoder.get_embedding(test_string)
            
            # Load saved encoder
            new_encoder = EncoderForInference.from_pretrained(save_path)
            
            # Test embedding after loading
            embedding_after = new_encoder.get_embedding(test_string)
            
            # Check embeddings are the same
            assert torch.allclose(embedding_before, embedding_after)
    finally:
        # Cleanup
        Path(checkpoint_path).unlink()

def test_batch_processing():
    # Create dummy checkpoint
    checkpoint_path = create_dummy_checkpoint()
    
    try:
        # Load encoder
        encoder = EncoderForInference.from_pretrained(checkpoint_path)
        
        # Create batch of test strings
        test_strings = [
            "First test string.",
            "Second test string, a bit longer.",
            "Third test string, even longer than the second one."
        ]
        
        # Get embeddings for each string
        embeddings = [encoder.get_embedding(s) for s in test_strings]
        
        # Check embeddings
        assert len(embeddings) == len(test_strings)
        for emb in embeddings:
            assert isinstance(emb, torch.Tensor)
            assert emb.shape == (1, 512)  # batch_size=1, embedding_dim=512
            assert not torch.isnan(emb).any()
        
        # Check embeddings are different (high probability they should be)
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                assert not torch.allclose(embeddings[i], embeddings[j])
    finally:
        # Cleanup
        Path(checkpoint_path).unlink() 