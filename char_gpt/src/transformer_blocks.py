# transformer_blocks.py
# ---------------------------------------------
# Step 13A: Simple Self-Attention Head (single head)
# ---------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, block_size):
        super().__init__()

        # Linear layers to produce Query, Key, Value
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.key   = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)

        # Causal mask (so GPT can't look ahead)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape   # batch, time, channels

        # Produce Q, K, V matrices
        Q = self.query(x)  # (B, T, head_dim)
        K = self.key(x)    # (B, T, head_dim)
        V = self.value(x)  # (B, T, head_dim)

        # Compute attention scores: Q * K^T
        scores = Q @ K.transpose(-2, -1) / (K.size(-1) ** 0.5)

        # Apply mask to prevent looking forward
        mask = self.mask[:T, :T]
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        attn = F.softmax(scores, dim=-1)

        attn = self.dropout(attn)

        # Weighted sum of values
        out = attn @ V  # (B, T, head_dim)

        return out



# -----------------------------------------------------
# Step 13B: Multi-Head Self Attention
# -----------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim, block_size):
        super().__init__()

        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_dim, head_dim, block_size)
            for _ in range(num_heads)
        ])

        # Final linear layer to mix all head outputs
        self.proj = nn.Linear(num_heads * head_dim, embed_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Apply all heads and concatenate their outputs along the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # Final projection back to embed_dim
        out = self.proj(out)

        return self.dropout(out)



# -----------------------------------------------------
# Step 13C: Feedforward Network (FFN)
# -----------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        # Hidden size is usually 4x larger (GPT style)
        hidden_dim = embed_dim * 4

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),                 # non-linear activation
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)




# -----------------------------------------------------
# Step 13D: Full Transformer Block
# -----------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim, block_size):
        super().__init__()

        # Components
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.attn = MultiHeadAttention(embed_dim, num_heads, head_dim, block_size)
        self.ffn  = FeedForward(embed_dim)

    def forward(self, x):
        # 1) Attention with residual
        x = x + self.attn(self.ln1(x))

        # 2) Feedforward with residual
        x = x + self.ffn(self.ln2(x))

        return x
