from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

class Executor:
    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        """
        Initialize the executor with a QA model.
        
        Args:
            model_name: Name of the HuggingFace QA model to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        
    def execute_query(self, query: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a query on a single item.
        
        Args:
            query: The question to answer
            item: The data item (can be text or tabular)
            
        Returns:
            Dict containing the answer and confidence score
        """
        # Determine item type and use appropriate method
        if isinstance(item.get("content"), pd.DataFrame):
            return self._execute_table_query(query, item)
        else:
            return self._execute_text_query(query, item)
            
    def _execute_text_query(self, query: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a query on a text item using BERT QA."""
        # Get the context from the item
        context = item["content"]
        
        # Tokenize
        inputs = self.tokenizer(
            query,
            context,
            max_length=512,
            truncation=True,
            stride=128,
            padding=True,
            return_tensors="pt"
        )

        # Remove offset_mapping as it's not needed for the model
        if 'offset_mapping' in inputs:
            del inputs['offset_mapping']
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get the most likely answer span
        start_logits = outputs.start_logits[0].cpu().numpy()
        end_logits = outputs.end_logits[0].cpu().numpy()
        
        # Get the most likely answer span
        answer_start = np.argmax(start_logits)
        answer_end = np.argmax(end_logits[answer_start:]) + answer_start
        
        # Convert tokens back to text
        answer = self.tokenizer.decode(
            inputs["input_ids"][0][answer_start:answer_end + 1],
            skip_special_tokens=True
        )
        
        # Calculate confidence score
        confidence = float(
            torch.softmax(torch.tensor(start_logits), dim=0)[answer_start] *
            torch.softmax(torch.tensor(end_logits), dim=0)[answer_end]
        )
        
        return {
            "answer": answer,
            "confidence": confidence,
            "source_type": "text",
            "source": item
        }
        
    def _execute_table_query(self, query: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a query on a table using a simple neural approach.
        For now, we'll convert the table row-by-row to text and use the same QA model.
        """
        df = item["content"]
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Table content must be a pandas DataFrame")
            
        # Convert table to text format row by row
        rows_text = []
        for idx, row in df.iterrows():
            row_text = " ".join(f"{col}: {val}" for col, val in row.items())
            rows_text.append(row_text)
            
        # Combine all rows into a single context
        context = " | ".join(rows_text)
        
        # Create a temporary text item
        text_item = {
            "content": context,
            "type": "table",
            "original_table": df
        }
        
        # Use text QA to find the answer
        result = self._execute_text_query(query, text_item)
        result["source_type"] = "table"
        
        return result 