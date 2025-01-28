import torch
import pytest
from symphony.embeddings.model import SymphonyAutoEncoder

def test_symphony_autoencoder_forward():
    # Initialize model with T5-small's hidden size
    model = SymphonyAutoEncoder(embedding_dim=512)
    
    # Create dummy input data
    batch_size = 2
    seq_length = 10
    vocab_size = model.t5.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Test forward pass without labels
    logits = model(input_ids, attention_mask)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, seq_length, vocab_size)
    
    # Test forward pass with labels
    labels = torch.randint(0, vocab_size, (batch_size, seq_length))
    loss, logits = model(
        input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()  # scalar
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, seq_length, vocab_size)

def test_symphony_autoencoder_encode():
    # Initialize model
    model = SymphonyAutoEncoder(embedding_dim=512)
    
    # Create dummy input data
    batch_size = 2
    seq_length = 10
    vocab_size = model.t5.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Test encode
    embedded = model.encode(input_ids, attention_mask)
    assert isinstance(embedded, torch.Tensor)
    assert embedded.shape == (batch_size, 512)  # embedding_dim = 512

def test_symphony_autoencoder_decode():
    # Initialize model
    model = SymphonyAutoEncoder(embedding_dim=512)
    
    # Create dummy input data
    batch_size = 2
    seq_length = 10
    vocab_size = model.t5.config.vocab_size
    embedded = torch.randn(batch_size, 512)  # Match T5-small hidden size
    decoder_input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    decoder_attention_mask = torch.ones(batch_size, seq_length)
    
    # Test decode
    logits = model.decode(embedded, decoder_input_ids, decoder_attention_mask)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, seq_length, vocab_size)
