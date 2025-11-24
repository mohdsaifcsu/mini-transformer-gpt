import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """
    Dataset for GPT-2 BPE text.
    Produces (input_ids, target_ids) for autoregressive training.
    """

    def __init__(self, text, tokenizer, block_size=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Encode entire corpus once
        print("Encoding full dataset with GPT-2 BPE tokenizer")
        self.data = tokenizer.encode(text)

        print(f"Total tokens: {len(self.data)}")

    def __len__(self):
        # Number of valid training windows.
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]

        input_ids  = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:],  dtype=torch.long)

        return input_ids, target_ids
