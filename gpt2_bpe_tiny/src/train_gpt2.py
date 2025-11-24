# train_gpt2.py
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import re


from gpt2_bpe_tiny.src.bpe_tokenizer import GPT2BPETokenizer
from gpt2_bpe_tiny.src.dataset import TextDataset
from gpt2_bpe_tiny.src.gpt2_model import GPT2Tiny



def load_latest_checkpoint():

    ckpt_dir = "gpt2_bpe_tiny/checkpoints"

    files = [f for f in os.listdir(ckpt_dir) if f.startswith("model_epoch")]

    if not files:
        raise FileNotFoundError("No checkpoint found in gpt2_bpe_tiny/checkpoints")

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
        vocab_file="gpt2_bpe_tiny/vocab/vocab.json",
        merges_file="gpt2_bpe_tiny/vocab/merges.txt"
    )

    # ------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------
    print("Loading dataset.....")
    text = open("gpt2_bpe_tiny/data/tiny_shakespeare.txt", "r", encoding="utf-8").read()

    block_size = 64
    dataset = TextDataset(text, tokenizer, block_size=block_size)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # ------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT2Tiny(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        embed_dim=128,
        num_layers=4,
        num_heads=4
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # ------------------------------------------------------------
    # 4. Optimizer
    # ------------------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    # ------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------
    epochs = 2
    print("Training started.....")

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
                print(f"Epoch {epoch+1}, Step {i+1}, Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        save_path = f"gpt2_bpe_tiny/checkpoints/model_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

    print("Training finished!")


if __name__ == "__main__":
    main()
