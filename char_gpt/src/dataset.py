# dataset.py
# Creates training sequences for next token prediction
# -----------------------------------------------------

import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size=64):
        """
        text: raw string
        tokenizer: CharTokenizer instance
        block_size: length of each training example
        """
        self.tokenizer = tokenizer
        self.block_size = block_size

        # convert entire text to token IDs
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        # Number of sequences we can extract
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Input: first block_size tokens
        x = self.data[idx:idx + self.block_size]

        # Target: next block_size tokens (shifted by 1)
        y = self.data[idx + 1:idx + 1 + self.block_size]

        return x, y
