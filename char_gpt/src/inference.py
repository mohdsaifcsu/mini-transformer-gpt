# inference.py
# ----------------------------------------
# Generate text using the trained tiny GPT
# ----------------------------------------

import torch
from src.tokenizer import CharTokenizer
from src.gpt_model import GPTModel


def generate(model, tokenizer, start_text, max_len=100, temperature=1.0, top_k=None, top_p=None):
    model.eval()

    idx = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long)

    for _ in range(max_len):

        # Ensure we do not exceed block_size
        if idx.size(1) > model.block_size:
            idx_cond = idx[:, -model.block_size:]
        else:
            idx_cond = idx

        # Forward pass
        logits = model(idx_cond)

        # Get last-step logits
        logits = logits[:, -1, :]  
        logits = logits / temperature

        # ----- TOP-K sampling -----
        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            min_value = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_value, torch.full_like(logits, float('-inf')), logits)

        # ----- TOP-P (nucleus) sampling -----
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)

            # Mask tokens outside top_p
            mask = cumulative_probs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False

            sorted_logits[mask] = float('-inf')
            logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Sample token
        next_token = torch.multinomial(probs, num_samples=1)

        # Append it
        idx = torch.cat([idx, next_token], dim=1)

    return tokenizer.decode(idx[0].tolist())


def main():
    # Load the text to rebuild tokenizer
    text = open("data/tiny_corpus.txt").read()
    tokenizer = CharTokenizer(text)

    vocab_size = tokenizer.vocab_size
    embed_dim = 64
    block_size = 32
    num_heads = 4
    num_layers = 2

    # Load model
    model = GPTModel(vocab_size, embed_dim, block_size, num_heads, num_layers)
    model.load_state_dict(torch.load("gpt_tiny.pth"))
    print("Model loaded!")

    # Generate text
    output = generate(
        model,
        tokenizer,
        start_text="h",
        max_len=60,
        temperature=0.8,
        top_k=10,
        top_p=0.9,
    )

    print("\nGenerated text:\n")
    print(output)


if __name__ == "__main__":
    main()
