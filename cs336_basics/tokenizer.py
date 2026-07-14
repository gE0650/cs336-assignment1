from collections.abc import Iterable, Iterator
import json
import regex as re
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

        if self.special_tokens:
            escaped_tokens = [re.escape(token) for token in self.special_tokens]
            escaped_tokens.sort(key=len, reverse=True)
            separator_pattern = "(" + "|".join(escaped_tokens) + ")"
            text_parts = re.split(separator_pattern, text)
        else:
            text_parts = [text]

        ids = []
        #tokens_bytes = []
        bytes_id = {token: id for (id, token) in self.vocab.items()}

        for text_part in text_parts:
            if self.special_tokens and (text_part in self.special_tokens):
                ids.append(bytes_id[text_part.encode("utf-8")])
            else:
                for match in re.finditer(PAT, text_part):
                    token_bytes = []
                    for letter in match.group(0):
                        letter_bytes = letter.encode("utf-8")
                        pieces = [bytes([byte_value]) for byte_value in letter_bytes]
                        token_bytes += pieces
        
                    for merge in self.merges:
                        pairs = list(zip(token_bytes, token_bytes[1:]))
                        i = 0
                        while i < len(pairs):
                            pair = pairs[i]
                            if pair == merge:
                                token_bytes[i] = token_bytes[i] + token_bytes[i + 1]
                                del token_bytes[i + 1]
                                pairs = list(zip(token_bytes, token_bytes[1:]))
                            i += 1
                
                    ids.extend([bytes_id[token] for token in token_bytes])

        
        
        return ids
            
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            encoded_chunk = (self.encode(chunk))
            for id in encoded_chunk:
                yield id
        
        


    def decode(self, ids: list[int]) -> str:
        all_bytes = bytes()
        for id in ids:
            all_bytes += self.vocab[id]
        
        return all_bytes.decode("utf-8", errors="replace")