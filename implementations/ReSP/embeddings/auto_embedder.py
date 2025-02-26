import numpy as np
import pandas as pd
import openai
import tiktoken
from typing import List, Union, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class AutoEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", max_tokens: int = 8192):
        """
        Initialize the embedder with OpenAI API.
        
        Args:
            api_key: OpenAI API key
            model: Name of the OpenAI embedding model to use.
                  Default is text-embedding-3-small which offers good balance of performance and cost.
            max_tokens: Maximum number of tokens allowed per input
        """
        openai.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        # text-embedding-3-small has 1536 dimensions
        self.embedding_dim = 1536 if "text-embedding-3" in model else 1536
        
        # Initialize tokenizer for token counting
        self.tokenizer = tiktoken.encoding_for_model(model) if model.startswith("text-embedding-3") else tiktoken.get_encoding("cl100k_base")
        
        # Cost optimization settings
        self.max_batch_size = 128  # Increased from default value
        self.token_utilization_factor = 0.95  # Use 95% of available tokens

    def _truncate_text(self, text: str) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            
        Returns:
            str: Truncated text
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.max_tokens:
            return text
            
        # Truncate to max tokens and decode back to text
        truncated_tokens = tokens[:self.max_tokens]
        return self.tokenizer.decode(truncated_tokens)

    def _batch_texts(self, texts: List[str], batch_size: int = 64) -> List[List[str]]:
        """
        Create batches of texts, ensuring each batch doesn't exceed token limits but
        maximizing token utilization to reduce API calls.
        
        Args:
            texts: List of texts to batch
            batch_size: Maximum batch size (default increased to 64)
            
        Returns:
            List[List[str]]: Batches of texts
        """
        # Use the instance max_batch_size if provided batch_size is smaller
        batch_size = max(batch_size, self.max_batch_size)
        
        # Calculate max tokens per batch (OpenAI limit is 8192 for embedding-3-small)
        max_batch_tokens = min(self.max_tokens * 8, 8192)  # Ensure we don't exceed API limits
        
        # First, preprocess all texts to get token counts
        text_items = []
        for text in texts:
            truncated_text = self._truncate_text(text)
            token_count = len(self.tokenizer.encode(truncated_text))
            text_items.append((truncated_text, token_count))
        
        # Sort by token count (largest first) for better packing
        text_items.sort(key=lambda x: x[1], reverse=True)
        
        batches = []
        current_batch = []
        current_batch_tokens = 0
        
        # Process texts from largest to smallest for efficient packing
        for text, token_count in text_items:
            # If this item alone is too large, it needs its own batch
            if token_count > max_batch_tokens:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_tokens = 0
                
                # Add large item as its own batch
                batches.append([text])
                continue
                
            # If adding this text would exceed batch size or token limit, start new batch
            if (len(current_batch) >= batch_size or 
                current_batch_tokens + token_count > max_batch_tokens * self.token_utilization_factor):
                if current_batch:
                    batches.append(current_batch)
                current_batch = [text]
                current_batch_tokens = token_count
            else:
                current_batch.append(text)
                current_batch_tokens += token_count
        
        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)
            
        # Log batch efficiency information
        total_items = len(texts)
        avg_batch_size = total_items / max(1, len(batches))
        print(f"Created {len(batches)} batches for {total_items} items (avg: {avg_batch_size:.1f} items/batch)")
        
        return batches

    @retry(stop=stop_after_attempt(3), 
           wait=wait_exponential(multiplier=1, min=4, max=10),
           retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError, openai.RateLimitError)))
    def _embed_batch(self, batch: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            batch: List of texts to embed
            
        Returns:
            np.ndarray: Embedding vectors
        """
        try:
            # Generate embeddings using OpenAI API
            response = openai.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float"  # Explicitly request float format
            )
            
            # Extract embeddings from response
            embeddings = np.array([item.embedding for item in response.data])
            return embeddings
        except Exception as e:
            print(f"Error embedding batch of {len(batch)} items: {e}")
            # If batch size is 1, we can't reduce further, so raise
            if len(batch) == 1:
                raise
                
            # If batch has multiple items and error is a 400 (context length exceeded)
            # Try splitting the batch in half
            if "400" in str(e) and len(batch) > 1:
                print(f"Splitting batch of size {len(batch)} into two smaller batches")
                mid = len(batch) // 2
                first_half = self._embed_batch(batch[:mid])
                second_half = self._embed_batch(batch[mid:])
                return np.vstack([first_half, second_half])
                
            # For other errors, try embedding one by one
            print(f"Retrying with individual embedding for batch of size {len(batch)}")
            results = []
            for text in batch:
                try:
                    result = self._embed_batch([text])
                    results.append(result)
                except Exception as inner_e:
                    print(f"Error embedding single item: {inner_e}")
                    # Use a zero vector as fallback
                    results.append(np.zeros((1, self.embedding_dim)))
            
            return np.vstack(results)

    def embed_text(self, text: Union[str, List[str]], batch_size: int = 64) -> np.ndarray:
        """
        Generate embeddings for input text using OpenAI's embedding model.
        
        Args:
            text: Single string or list of strings to embed
            batch_size: Size of batches to process (increased default to 64)
            
        Returns:
            np.ndarray: Embedding vectors with shape (n_samples, embedding_dim)
        """
        if isinstance(text, str):
            text = [text]
        
        # Empty input check
        if not text or len(text) == 0:
            return np.zeros((0, self.embedding_dim))
            
        # Track total cost estimation
        total_tokens = sum(len(self.tokenizer.encode(t)) for t in text)
        estimated_cost = (total_tokens / 1000000) * 0.02  # $0.02 per million tokens
        print(f"Embedding {len(text)} items with ~{total_tokens} tokens (est. cost: ${estimated_cost:.4f})")
        
        # Create batches with optimized batch size
        batches = self._batch_texts(text, batch_size)
        
        # Process each batch
        all_embeddings = []
        for i, batch in enumerate(batches):
            batch_tokens = sum(len(self.tokenizer.encode(t)) for t in batch)
            print(f"Processing batch {i+1}/{len(batches)} with {len(batch)} items ({batch_tokens} tokens)")
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.append(batch_embeddings)
        
        # Combine results
        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            return np.zeros((len(text), self.embedding_dim))

    def embed_dataset(self, dataset, batch_size: int = 64) -> np.ndarray:
        """
        Embed an entire dataset with proper batching.
        
        Args:
            dataset: Dataset object with __iter__ method returning text representations
            batch_size: Batch size for processing (increased default)
            
        Returns:
            np.ndarray: Embeddings for all items
        """
        print(f"Preparing to embed dataset with {len(dataset)} items")
        
        # Collect all items first to allow for optimal batching
        all_items = []
        for item in dataset:
            all_items.append(item)
        
        # Embed all items with optimized batching
        return self.embed_text(all_items, batch_size=batch_size)

    def embed_tabular(self, df: pd.DataFrame, columns: List[str] = None) -> np.ndarray:
        """
        Generate embeddings for tabular data by converting rows to text.
        
        Args:
            df: Pandas DataFrame to embed
            columns: List of column names to include. If None, uses all columns.
            
        Returns:
            np.ndarray: Embedding vectors with shape (n_rows, embedding_dim)
        """
        if columns is None:
            columns = df.columns.tolist()
            
        # Convert each row to a text representation
        text_rows = []
        for _, row in df.iterrows():
            row_text = " ".join(f"{col}: {row[col]}" for col in columns)
            text_rows.append(row_text)
            
        # Generate embeddings for the text representations
        return self.embed_text(text_rows)

    def compute_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings with shape (n1, embedding_dim)
            embeddings2: Second set of embeddings with shape (n2, embedding_dim)
            
        Returns:
            np.ndarray: Similarity matrix with shape (n1, n2)
        """
        # Normalize the embeddings
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        embeddings1_norm = embeddings1 / norm1
        embeddings2_norm = embeddings2 / norm2
        
        # Compute cosine similarity
        return np.dot(embeddings1_norm, embeddings2_norm.T)