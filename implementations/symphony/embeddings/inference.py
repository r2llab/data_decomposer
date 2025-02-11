import torch
import torch.nn as nn
from typing import Optional
from transformers import T5Tokenizer
from pathlib import Path

from .model import SymphonyAutoEncoder

class EncoderForInference(nn.Module):
    """Encoder part of SymphonyAutoEncoder for inference."""
    
    def __init__(
        self,
        encoder,
        projection: nn.Linear,
        tokenizer: T5Tokenizer,
        max_length: int = 512,
        device: Optional[str] = None
    ):
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.to(self.device)
        
        # Set to evaluation mode
        self.eval()
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        tokenizer_name: str = "t5-small",
        device: Optional[str] = None
    ) -> "EncoderForInference":
        """Load encoder from a trained checkpoint.
        
        Supports both full model checkpoints and encoder-only checkpoints.
        """
        try:
            # Try loading as a safe checkpoint first
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                pickle_module=torch.serialization.pickle
            )
        except Exception:
            # Fallback to regular loading if needed
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Initialize tokenizer
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        
        # Create a new model instance
        full_model = SymphonyAutoEncoder()
        
        # Handle both full model and encoder-only checkpoints
        if "model_state_dict" in checkpoint:
            # Full model checkpoint
            full_model.load_state_dict(checkpoint["model_state_dict"])
            encoder = full_model.t5.encoder
            projection = full_model.projection
        else:
            # Encoder-only checkpoint
            encoder = full_model.t5.encoder
            projection = full_model.projection
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
            projection.load_state_dict(checkpoint["projection_state_dict"])
        
        # Create encoder-only model
        return cls(
            encoder=encoder,
            projection=projection,
            tokenizer=tokenizer,
            device=device
        )
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get embeddings for input tokens."""
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Get the mean of all token embeddings as the sequence representation
        if attention_mask is not None:
            # Create mask for mean calculation
            mask = attention_mask.unsqueeze(-1).float()
            sequence_output = (encoder_outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            sequence_output = encoder_outputs.last_hidden_state.mean(dim=1)
        
        # Project to fixed size
        embeddings = self.projection(sequence_output)
        return embeddings
    
    @torch.no_grad()
    def get_embedding(self, input_string: str) -> torch.Tensor:
        """Get embedding for a single input string."""
        # Tokenize input
        encoded = self.tokenizer(
            input_string,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Get embedding
        embedding = self.forward(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask
        )
        
        return embedding
    
    def save_pretrained(self, save_path: str):
        """Save the encoder model to disk."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state using safe serialization
        torch.save(
            {
                "encoder_state_dict": self.encoder.state_dict(),
                "projection_state_dict": self.projection.state_dict(),
            },
            save_path,
            pickle_module=torch.serialization.pickle  # Use safe serialization
        )
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path.parent / "tokenizer") 