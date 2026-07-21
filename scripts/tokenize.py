from cs336_basics.tokenizer import Tokenizer
import argparse, torch
import numpy as np

parser = argparse.ArgumentParser(description="Tokenizer a file")
parser.add_argument("--file-input", required=True, help="raw text path")
parser.add_argument("--tokenizer-input", required=True, help="tokenizer data path")
parser.add_argument("--output", required=True, help="output .npy path")
args = parser.parse_args()

file_input_path = args.file_input
tokenizer_input_path = args.tokenizer_input
output_path = args.output

# create Tokenizer instance

tokenizer_data = torch.load(tokenizer_input_path)
merges = tokenizer_data["merges"]
vocab = tokenizer_data["vocab"]
special_tokens = tokenizer_data["special_tokens"]

tokenizer = Tokenizer(vocab, merges, special_tokens)

# naive implementation, need to switch to stream
text = open(file_input_path, "r", encoding="utf-8").read()
tokenized_text = tokenizer.encode(text)

np.save(output_path, np.array(tokenized_text, dtype=np.uint16))