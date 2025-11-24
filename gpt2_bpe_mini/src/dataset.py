# dataset.py  (GPT-2 BPE Dataset for MINI model)

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    GPT-2 BPE dataset for autoregressive training.
    Converts full text → token IDs → sliding windows.
    """

    def __init__(self, text, tokenizer, block_size=128):
        super().__init__()

        self.tokenizer = tokenizer
        self.block_size = block_size

        # ------------------------------------------------------------
        # Encode full dataset once (BPE tokenization)
        # ------------------------------------------------------------
        print("Encoding full dataset with GPT-2 BPE tokenizer...")
        self.data = tokenizer.encode(text)

        print(f"Total tokens in dataset: {len(self.data)}")

    def __len__(self):
        # number of training samples = all possible windows
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        """
        Returns:
            input_ids  : first  block_size tokens
            target_ids : next   block_size tokens
            (classic autoregressive LM)
        """
        chunk = self.data[idx : idx + self.block_size + 1]

        input_ids  = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:],  dtype=torch.long)

        return input_ids, target_ids
