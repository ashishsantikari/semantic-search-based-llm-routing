"""
Embedding service wrapping HuggingFace transformers directly.

Provides a singleton-style EmbeddingService class that loads a
sentence-transformer model via AutoModel / AutoTokenizer and computes
mean-pooled L2-normalized embeddings on CPU.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.config import EMBEDDING_MODEL_NAME


class EmbeddingService:
    """Computes mean-pooled L2-normalized embeddings using a HuggingFace model."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(self.device)
        self.model.eval()

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string into a 384-dim L2-normalized vector."""
        if not text:
            return np.zeros(384, dtype=np.float32)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        token_embeddings = outputs.last_hidden_state  # (1, seq_len, 384)
        attention_mask = inputs["attention_mask"]  # (1, seq_len)

        # Mean pooling
        mask = attention_mask.unsqueeze(-1).float()  # (1, seq_len, 1)
        masked_embeddings = token_embeddings * mask
        summed = masked_embeddings.sum(dim=1)  # (1, 384)
        counts = mask.sum(dim=1)  # (1, 1)
        mean_pooled = summed / counts

        # L2 normalize
        norm = torch.norm(mean_pooled, p=2, dim=1, keepdim=True)
        normalized = mean_pooled / norm

        return normalized.squeeze(0).cpu().numpy()
