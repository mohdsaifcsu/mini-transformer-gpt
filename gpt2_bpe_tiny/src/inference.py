# inference.py

import torch

# -----------------------------
# Updated imports for new folder
# -----------------------------
from gpt2_bpe_tiny.src.bpe_tokenizer import GPT2BPETokenizer
from gpt2_bpe_tiny.src.gpt2_model import GPT2Tiny
from gpt2_bpe_tiny.src.train_gpt2 import load_latest_checkpoint
from gpt2_bpe_tiny.src.generate import generate


def main():
    device = "cpu"

    # -----------------------------
    # Load tokenizer
    # -----------------------------
    tokenizer = GPT2BPETokenizer(
        vocab_file="gpt2_bpe_tiny/vocab/vocab.json",
        merges_file="gpt2_bpe_tiny/vocab/merges.txt"
    )

    # -----------------------------
    # Load GPT-2 model
    # -----------------------------
    model = GPT2Tiny(
        vocab_size=tokenizer.vocab_size,
        block_size=64,
        embed_dim=128,
        num_layers=4,
        num_heads=4
    ).to(device)

    # -----------------------------
    # Load latest checkpoint
    # -----------------------------
    ckpt_path = load_latest_checkpoint()
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint: {ckpt_path}")

    # -----------------------------
    # Prompt
    # -----------------------------
    prompt = input("\nEnter prompt: ").strip()

    print("\nGenerating text...\n")

    # -----------------------------
    # Generate text
    # -----------------------------
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=200,
        temperature=0.9,
        top_k=50,
        top_p=0.95
    )

    print("\n========= RESULT =========\n")
    print(output)
    print("\n==========================\n")


if __name__ == "__main__":
    main()
