# train_gpt.py (CHAR-LEVEL GPT)
# ----------------------------------------------
# - Correct paths for new folder structure
# - Saves checkpoints in char_gpt/checkpoints/
# - Saves final model in char_gpt/checkpoints/
# - Loads data from char_gpt/data/
# - Imports from char_gpt.src.*
# ----------------------------------------------

import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from tqdm import tqdm

# Correct imports for new structure
from char_gpt.src.tokenizer import CharTokenizer
from char_gpt.src.dataset import TextDataset
from char_gpt.src.gpt_model import GPTModel


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main():
    # -------------------------
    # 1) Training hyperparams
    # -------------------------
    block_size = 32
    batch_size = 32
    num_epochs = 5
    learning_rate = 3e-4

    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # -------------------------
    # 2) Load text and tokenizer
    # -------------------------
    data_path = "char_gpt/data/tiny_corpus.txt"
    text = open(data_path, "r").read()

    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # -------------------------
    # 3) Dataset split
    # -------------------------
    full_dataset = TextDataset(text, tokenizer, block_size=block_size)
    dataset_len = len(full_dataset)
    train_len = int(0.9 * dataset_len)
    val_len = dataset_len - train_len

    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
    print(f"Dataset size: {dataset_len} (train={train_len}, val={val_len})")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------
    # 4) Model
    # -------------------------
    embed_dim = 64
    num_heads = 4
    num_layers = 2

    model = GPTModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        block_size=block_size,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # -------------------------
    # 5) Optimizer
    # -------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # -------------------------
    # 6) Ensure checkpoints directory exists
    # -------------------------
    ckpt_dir = "char_gpt/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")

    # -------------------------
    # 7) Training
    # -------------------------
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for x, y in loop:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            loop.set_postfix(loss=loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)

        # -------------------------
        # 8) Validation
        # -------------------------
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                val_loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    y.view(-1),
                )

                val_losses.append(val_loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --------------------------------
        # 9) Save BEST model checkpoint
        # --------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best model saved to {ckpt_path}")

    # --------------------------------
    # 10) Save final model
    # --------------------------------
    final_model_path = os.path.join(ckpt_dir, "gpt_tiny.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved as {final_model_path}")


if __name__ == "__main__":
    main()
