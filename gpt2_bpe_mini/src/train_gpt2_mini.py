# train_gpt2_mini.py
# Training script for GPT-2 MINI (8-layer)

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import re

# ------------------------------------------------------------
# Correct MINI imports
# ------------------------------------------------------------
from gpt2_bpe_mini.src.bpe_tokenizer import GPT2BPETokenizer
from gpt2_bpe_mini.src.dataset import TextDataset
from gpt2_bpe_mini.src.gpt2_model_mini import GPT2Mini


# ================================================================
# Load latest checkpoint from gpt2_bpe_mini/checkpoints
# ================================================================
def load_latest_checkpoint():

    ckpt_dir = "gpt2_bpe_mini/checkpoints"

    files = [f for f in os.listdir(ckpt_dir) if f.startswith("model_epoch")]

    if not files:
        raise FileNotFoundError("No checkpoint found in gpt2_bpe_mini/checkpoints")

    def extract_epoch(filename):
        m = re.search(r"epoch(\d+)", filename)
        return int(m.group(1)) if m else -1

    files = sorted(files, key=lambda f: extract_epoch(f))
    latest = files[-1]

    return os.path.join(ckpt_dir, latest)



# ================================================================
# MAIN TRAINING SCRIPT
# ================================================================
def main():

    # ------------------------------------------------------------
    # 1. Load tokenizer
    # ------------------------------------------------------------
    tokenizer = GPT2BPETokenizer(
        vocab_file="gpt2_bpe_mini/vocab/vocab.json",
        merges_file="gpt2_bpe_mini/vocab/merges.txt"
    )

    # ------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------
    print("Loading dataset...")
    text_path = "gpt2_bpe_mini/data/tiny_shakespeare.txt"
    text = open(text_path, "r", encoding="utf-8").read()

    block_size = 128   # larger for MINI
    dataset = TextDataset(text, tokenizer, block_size=block_size)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # ------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT2Mini(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        embed_dim=256,    # MINI dims
        num_layers=8,
        num_heads=8
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------
    # 4. Optimizer
    # ------------------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    # ------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------
    epochs = 2
    print("Training started...")

    for epoch in range(epochs):
        running_loss = 0.0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 20 == 0:
                print(f"[Epoch {epoch+1}] Step {i+1} — Loss: {loss.item():.4f}")

        # epoch summary
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        save_path = f"gpt2_bpe_mini/checkpoints/model_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint: {save_path}")

    print("\nTraining finished!")


if __name__ == "__main__":
    main()
