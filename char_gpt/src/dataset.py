# dataset.py
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.tokens = tokenizer.encode(text)

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx+self.block_size+1]
        x = torch.tensor(chunk[:-1])
        y = torch.tensor(chunk[1:])
        return x, y
