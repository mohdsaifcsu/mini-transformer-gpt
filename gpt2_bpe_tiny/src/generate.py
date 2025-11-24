# generate.py
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=200,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
):
    """
    GPT-2 style text generation.
    """

    device = next(model.parameters()).device

    # Encode prompt -> initial token IDs
    ids = tokenizer.encode(prompt)
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    model.eval()

    for _ in range(max_new_tokens):

        # Only keep last block_size tokens
        idx_cond = ids[:, -model.block_size:]

        # Model forward pass
        logits = model(idx_cond)  # shape: (1, T, vocab_size)
        logits = logits[:, -1, :] / max(temperature, 1e-8)  # last step logits

        # -----------------------------
        # Top-k filtering
        # -----------------------------
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = -1e10

        # -----------------------------
        # Softmax -> probabilities
        # -----------------------------
        probs = F.softmax(logits, dim=-1)

        # -----------------------------
        # Top-p (nucleus sampling)
        # -----------------------------
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)

            mask = cumulative > top_p
            sorted_probs[mask] = 0

            # re-normalize
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(sorted_probs, 1)
            next_id = sorted_idx.gather(-1, next_token)
        else:
            next_id = torch.multinomial(probs, 1)

        # Append new token
        ids = torch.cat([ids, next_id], dim=1)

        # -----------------------------
        # Stop when end-of-text encountered
        # -----------------------------
        if next_id.item() == tokenizer.encoder.get("<|endoftext|>", -1):
            break

    # Decode tokens -> string
    return tokenizer.decode(ids[0].tolist())
