import torch
import torch.nn as nn
from transformers import T5Model, T5Config
from typing import Optional, Tuple, Union

class SymphonyAutoEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,  # Changed default to match T5-small's hidden size
        model_name: str = "t5-small",
        max_length: int = 512,
    ):
        super().__init__()
        # Initialize T5 model for encoder-decoder
        self.t5 = T5Model.from_pretrained(model_name)
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        
        # Projection layer to get fixed-size embeddings
        self.projection = nn.Linear(self.t5.config.hidden_size, embedding_dim)
        
        # Output projection to vocab size for generation
        self.output_projection = nn.Linear(self.t5.config.hidden_size, self.t5.config.vocab_size)
        
        # Initialize loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode input sequence to fixed-size embedding."""
        # Get encoder outputs
        encoder_outputs = self.t5.encoder(
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
        compressed = self.projection(sequence_output)
        return compressed

    def decode(
        self,
        embedded: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode from embedding back to sequence."""
        batch_size = embedded.size(0)
        sequence_length = decoder_input_ids.size(1)
        
        # Project embedding back to T5's hidden size if needed
        if embedded.size(-1) != self.t5.config.hidden_size:
            embedded = self.projection(embedded)
        
        # Repeat the embedding for each position in the sequence
        expanded_embedding = embedded.unsqueeze(1).expand(-1, sequence_length, -1)
        
        # Use T5 decoder
        decoder_outputs = self.t5.decoder(
            inputs_embeds=expanded_embedding,
            encoder_hidden_states=expanded_embedding,
            attention_mask=decoder_attention_mask,
        )
        
        # Project to vocabulary size
        logits = self.output_projection(decoder_outputs.last_hidden_state)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask for input
            decoder_input_ids: Input ids for decoder (for training)
            decoder_attention_mask: Attention mask for decoder
            labels: Target labels for computing loss
            
        Returns:
            If labels is provided:
                tuple of (loss, logits)
            else:
                logits
        """
        # 1. Encode to fixed-size embedding
        compressed = self.encode(input_ids, attention_mask)
        
        # 2. Decode
        if decoder_input_ids is None:
            decoder_input_ids = input_ids
            
        if decoder_attention_mask is None:
            decoder_attention_mask = attention_mask
            
        logits = self.decode(
            compressed,
            decoder_input_ids,
            decoder_attention_mask,
        )
        
        # 3. Calculate loss if labels provided
        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, self.t5.config.vocab_size),
                labels.view(-1),
            )
            return loss, logits
            
        return logits
