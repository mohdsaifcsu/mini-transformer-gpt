import json
import regex as re
from typing import List, Tuple

class GPT2BPETokenizer:
    def __init__(self, vocab_file: str, merges_file: str):
        """
        GPT-2 BPE Tokenizer (minimal clean version)
        """

        # ------------------------------------------------------------
        # Load vocabulary
        # ------------------------------------------------------------
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)

        # reverse mapping
        self.decoder = {v: k for k, v in self.encoder.items()}

        # vocab size for GPT model
        self.vocab_size = len(self.encoder)

        # UNK token
        self.unk_token = "<|unk|>"
        if self.unk_token not in self.encoder:
            self.encoder[self.unk_token] = len(self.encoder)
            self.decoder[self.encoder[self.unk_token]] = self.unk_token

        # ------------------------------------------------------------
        # Load merges
        # ------------------------------------------------------------
        merges = []
        with open(merges_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append(tuple(parts))

        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

        # ------------------------------------------------------------
        # GPT-2 regex
        # ------------------------------------------------------------
        # using unicode categories via `regex` (not Python re)
        self.pat = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d"
            r"| ?\p{L}+| ?\p{N}+"
            r"| ?[^\s\p{L}\p{N}]+"
            r"|\s+(?!\S)|\s+"
        )

    # ================================================================
    # BPE MERGE LOGIC
    # ================================================================
    def get_pairs(self, word: List[str]):
        pairs = set()
        if len(word) < 2:
            return pairs
        prev = word[0]
        for ch in word[1:]:
            pairs.add((prev, ch))
            prev = ch
        return pairs

    def bpe(self, token: str) -> str:
        if token in self.encoder:
            return token

        word = list(token)
        pairs = self.get_pairs(word)

        if not pairs:
            return token

        while True:
            min_pair = None
            min_rank = float("inf")

            # find smallest rank merge pair
            for pair in pairs:
                if pair in self.bpe_ranks and self.bpe_ranks[pair] < min_rank:
                    min_rank = self.bpe_ranks[pair]
                    min_pair = pair

            if min_pair is None:
                break

            first, second = min_pair
            new_word = []
            i = 0

            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break

                new_word.extend(word[i:j])
                i = j

                # merge
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word
            pairs = self.get_pairs(word)

        return " ".join(word)

    # ================================================================
    # ENCODE
    # ================================================================
    def encode(self, text: str) -> List[int]:
        bpe_tokens = []
        tokens = re.findall(self.pat, text)

        for token in tokens:
            token = token.strip()
            if not token:
                continue

            bpe_token = self.bpe(token)

            for bt in bpe_token.split(" "):
                bpe_tokens.append(self.encoder.get(bt, self.encoder[self.unk_token]))

        return bpe_tokens

    # ================================================================
    # DECODE
    # ================================================================
    def decode(self, ids: List[int]) -> str:
        tokens = [self.decoder.get(i, self.unk_token) for i in ids]
        text = "".join(tokens)
        text = text.replace("Ġ", " ")  # GPT-2 space token
        return text
