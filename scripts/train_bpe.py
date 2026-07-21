from cs336_basics.train_bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer
import argparse, torch

parser = argparse.ArgumentParser(description="Train a tokenizer")
parser.add_argument("--input", required=True, help="raw text path")
parser.add_argument("--output", required=True, help="output tokenizer data path")
parser.add_argument("--vocab-size", type=int, default=10000)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()


input_path = args.input
output_path = args.output

vocab_size = args.vocab_size

special_tokens = ["<|endoftext|>"] # may need to modify

vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

res = {"vocab": vocab,
       "merges": merges,
       "special_tokens": special_tokens}

torch.save(res, output_path)