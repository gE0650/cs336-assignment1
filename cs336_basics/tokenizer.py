from collections.abc import Iterable, Iterator
import json
from tests.common import gpt2_bytes_to_unicode
from cs336_basics.train_bpe import pretoken_from_chunk, PAT

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with (
            open(vocab_filepath, "r", encoding="utf-8") as vocab_file,
            open(merges_filepath, "r", encoding="utf-8") as merges_file,
        ):
            
            vocab_rev = json.load(vocab_file)
            vocab = {}
            gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
            for token, id in vocab_rev.items():
                vocab[id] = bytes([gpt2_byte_decoder[letter] for letter in token])
            
            if special_tokens:
                for special_token in special_tokens:
                    if special_token.encode("utf-8") not in vocab.values():
                        vocab[len(vocab)] = special_token.encode("utf-8")

            merges = []
            
            for line in merges_file:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    list_str = cleaned_line.split(" ")
                    merges.append((bytes([gpt2_byte_decoder[letter] for letter in list_str[0]]), bytes([gpt2_byte_decoder[letter] for letter in list_str[1]])))

            return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        chunks = pretoken_from_chunk(PAT, text.encode("utf-8"), self.special_tokens)
        return [i for i in chunk for chunk in chunks]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        chunks = pretoken_from_chunk(PAT, text.encode("utf-8"), self.special_tokens)
        return [i for i in chunk for chunk in chunks]


    def decode(self, ids: list[int]) -> str:
        text = ""
        for id in ids:
            text += self.vocab[id].decode("utf-8")
        
        return text