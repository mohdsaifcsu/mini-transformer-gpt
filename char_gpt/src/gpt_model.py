# gpt_model.py
# ----------------------------------------------
# Minimal GPT model: embeddings + transformer blocks + LM head
# ----------------------------------------------

import torch
import torch.nn as nn
from src.transformer_blocks import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_layers):
        super().__init__()

        self.block_size = block_size

        # 1) Token Embeddings: convert token IDs -> vectors
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # 2) Positional Embeddings: learnable position vectors
        self.pos_embed = nn.Embedding(block_size, embed_dim)

        # 3) Transformer Blocks (stacked)
        head_dim = embed_dim // num_heads
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, head_dim, block_size)
            for _ in range(num_layers)
        ])

        # 4) LayerNorm before output
        self.ln_f = nn.LayerNorm(embed_dim)

        # 5) Output head: project embedding -> vocab size
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        """
        idx: (batch, time) of token IDs
        returns: logits of shape (batch, time, vocab_size)
        """

        B, T = idx.shape

        # Token embeddings
        tok_emb = self.token_embed(idx)                # (B, T, embed_dim)

        # Positional embeddings
        positions = torch.arange(T, device=idx.device)  # (T,)
        pos_emb = self.pos_embed(positions)             # (T, embed_dim)

        # Add both embeddings
        x = tok_emb + pos_emb                          # (B, T, embed_dim)

        # Apply each transformer block
        for block in self.blocks:
            x = block(x)

        # Final layernorm
        x = self.ln_f(x)

        # Output logits
        logits = self.head(x)                           # (B, T, vocab_size)

        return logits
