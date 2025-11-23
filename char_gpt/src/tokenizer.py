
# Simple Character Level Tokenizer for Tiny GPT
# --------------------------------------------------
# This tokenizer converts:
#    text -> list of token IDs
#    list of token IDs -> text
# --------------------------------------------------

class CharTokenizer:
    def __init__(self, text):
        """
        Builds a vocabulary from the given text.
        """
        self.vocab = sorted(list(set(text)))          # unique characters
        self.vocab_size = len(self.vocab)

        # char -> index mapping
        self.char_to_id = {ch: i for i, ch in enumerate(self.vocab)}

        # index -> char mapping
        self.id_to_char = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, text):
        """
        Convert text into a list of integers (token IDs).
        """
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids):
        """
        Convert a list of token IDs back into a string.
        """
        return ''.join([self.id_to_char[i] for i in ids])
