from cs336_basics.tokenizer import Tokenizer
import argparse, torch
import numpy as np
from itertools import islice

parser = argparse.ArgumentParser(description="Tokenizer a file")
parser.add_argument("--file-input", required=True, help="raw text path")
parser.add_argument("--tokenizer-input", required=True, help="tokenizer data path")
parser.add_argument("--output", required=True, help="output .bin path")
args = parser.parse_args()

file_input_path = args.file_input
tokenizer_input_path = args.tokenizer_input
output_path = args.output

# create Tokenizer instance

tokenizer_data = torch.load(f"results/tokenizer/{tokenizer_input_path}")
merges = tokenizer_data["merges"]
vocab = tokenizer_data["vocab"]
special_tokens = tokenizer_data["special_tokens"]

tokenizer = Tokenizer(vocab, merges, special_tokens)

# stream tokenize
with open(f"data/{file_input_path}", "r", encoding="utf-8") as file_in:
    with open(f"results/tokenized_text/{output_path}", "wb") as file_out:
        tokenize_res = tokenizer.encode_iterable(file_in)
        counter = 0
        chunk_size = 100_000 # hyperpamameter

        while True:
            buffer = np.array(list(islice(tokenize_res, chunk_size)), dtype=np.uint16)
            file_out.write(buffer.tobytes())
            if len(buffer) < chunk_size:
                break
            


