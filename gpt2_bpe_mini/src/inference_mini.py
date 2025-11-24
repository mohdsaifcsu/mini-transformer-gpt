# inference_mini.py
# Inference script for GPT-2 MINI (8-layer)

import torch

# ------------------------------------------------------------
# Correct MINI imports
# ------------------------------------------------------------
from gpt2_bpe_mini.src.bpe_tokenizer import GPT2BPETokenizer
from gpt2_bpe_mini.src.gpt2_model_mini import GPT2Mini
from gpt2_bpe_mini.src.train_gpt2_mini import load_latest_checkpoint
from gpt2_bpe_mini.src.generate_mini import generate_mini


def main():
    device = "cpu"  # (CPU is fine for inference)

    # ------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------
    tokenizer = GPT2BPETokenizer(
        vocab_file="gpt2_bpe_mini/vocab/vocab.json",
        merges_file="gpt2_bpe_mini/vocab/merges.txt"
    )

    # ------------------------------------------------------------
    # Load GPT-2 MINI model
    # ------------------------------------------------------------
    model = GPT2Mini(
        vocab_size=tokenizer.vocab_size,
        block_size=128,
        embed_dim=256,
        num_layers=8,
        num_heads=8
    ).to(device)

    # ------------------------------------------------------------
    # Load latest checkpoint
    # ------------------------------------------------------------
    ckpt_path = load_latest_checkpoint()
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"\nLoaded checkpoint: {ckpt_path}\n")

    # ------------------------------------------------------------
    # User input
    # ------------------------------------------------------------
    prompt = input("Enter your prompt: ").strip()

    print("\nGenerating text...\n")

    # ------------------------------------------------------------
    # Generate continuation
    # ------------------------------------------------------------
    output = generate_mini(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=200,
        temperature=0.9,
        top_k=50,
        top_p=0.95
    )

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------
    print("\n========= RESULT =========\n")
    print(output)
    print("\n==========================\n")


if __name__ == "__main__":
    main()
