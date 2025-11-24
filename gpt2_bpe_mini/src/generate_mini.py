# generate_mini.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def generate_mini(
    model,
    tokenizer,
    prompt,
    max_new_tokens=200,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
):
    device = next(model.parameters()).device

    # Encode prompt
    ids = tokenizer.encode(prompt)
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    model.eval()

    for _ in range(max_new_tokens):

        # Use only last block_size tokens
        idx_cond = ids[:, -model.block_size:]

        logits = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        # -------- SAFE LOGIT CLAMPING --------
        logits = torch.clamp(logits, -50, 50)

        # -------- TOP-K --------
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -1e10

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # -------- Prevent NaN --------
        if torch.isnan(probs).any():
            print("WARNING: softmax produced NaN, fixing...")
            probs = torch.nan_to_num(probs, nan=1e-9)

        # -------- TOP-P (nucleus) --------
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)

            mask = cumulative > top_p
            sorted_probs[mask] = 0

            total = sorted_probs.sum()

            if total == 0:
                sorted_probs = sorted_probs + 1e-9
                total = sorted_probs.sum()

            sorted_probs /= total

            next_token = torch.multinomial(sorted_probs, 1)
            next_id = sorted_idx.gather(-1, next_token)

        else:
            next_id = torch.multinomial(probs, 1)

        # Append token
        ids = torch.cat([ids, next_id], dim=1)

        # EOT (optional)
        if next_id.item() == tokenizer.encoder.get("<|endoftext|>", -1):
            break

    return tokenizer.decode(ids[0].tolist())
