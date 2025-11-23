# train_gpt.py
# ----------------------------------------
# Training loop for our tiny GPT model
# - GPU support
# - Train/validation split
# - Progress bar with tqdm
# - Gradient clipping
# - Checkpoint saving
# ----------------------------------------

import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from tqdm import tqdm

from src.tokenizer import CharTokenizer
from src.dataset import TextDataset
from src.gpt_model import GPTModel


def get_device():
    """Return GPU if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # for Apple Silicon
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main():
    # -----------------------------
    # 1) Basic config
    # -----------------------------
    block_size = 32
    batch_size = 32
    num_epochs = 5
    learning_rate = 3e-4

    # For reproducibility
    torch.manual_seed(42)

    device = get_device()
    print(f"Using device: {device}")

    # -----------------------------
    # 2) Load text & tokenizer
    # -----------------------------
    text = open("data/tiny_corpus.txt", "r").read()
    tokenizer = CharTokenizer(text)

    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # -----------------------------
    # 3) Create dataset & split
    # -----------------------------
    full_dataset = TextDataset(text, tokenizer, block_size=block_size)
    dataset_len = len(full_dataset)
    train_len = int(0.9 * dataset_len)
    val_len = dataset_len - train_len

    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

    print(f"Dataset size: {dataset_len} (train={train_len}, val={val_len})")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -----------------------------
    # 4) Build model
    # -----------------------------
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

    # -----------------------------
    # 5) Optimizer
    # -----------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # For saving best model
    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")

    # -----------------------------
    # 6) Training loop
    # -----------------------------
    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for x, y in loop:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)  # (B, T, V)

            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_losses.append(loss.item())
            loop.set_postfix(loss=loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)

        # -----------------------------
        # 7) Validation
        # -----------------------------
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

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # -----------------------------
        # 8) Save best checkpoint
        # -----------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join("checkpoints", "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best model saved to {ckpt_path}")

    # -----------------------------
    # 9) Save final model for inference
    # -----------------------------
    torch.save(model.state_dict(), "gpt_tiny.pth")
    print("Final model saved as gpt_tiny.pth")


if __name__ == "__main__":
    main()
