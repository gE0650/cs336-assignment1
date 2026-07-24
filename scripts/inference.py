
import argparse, torch
import numpy as np
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.base_modules import TransformerLM
from cs336_basics.training_modules import loadBatch, AdamW, crossEntrophy, gradClip, cosLRS, saveCheckPoint, loadCheckPoint

parser = argparse.ArgumentParser(description="Generating text")
parser.add_argument("--model", required=True, help="model path")
parser.add_argument("--output", required=True, help="output model data")
parser.add_argument("--prefix", required=True, help="output prefix path")
parser.add_argument("--tokenizer-input", required=True, help="tokenizer data path")
parser.add_argument("--max-length", required=True, help="max output length", type=int)

parser.add_argument("--vocab-size", required=True, help="vocabulary size", type=int)
parser.add_argument("--context-len", required=True, help="context length", type=int)
parser.add_argument("--d-model", required=True, help="d_model", type=int)
parser.add_argument("--num-layer", required=True, help="num_layer", type=int)
parser.add_argument("--num-heads", required=True, help="num_head", type=int)
parser.add_argument("--theta", required=True, help="rope theta", type=float)
parser.add_argument("--device", required=True, help="device")

parser.add_argument("--temperature", required=True, help="temperature", type=float)
parser.add_argument("--top-num", required=True, help="k for top_k", type=int)

args = parser.parse_args()

model_path = args.model
output_path = args.output
prefix_path = args.prefix
tokenizer_input_path = args.tokenizer_input
max_length = args.max_length

device = args.device
vocab_size = args.vocab_size
context_len = args.context_len
d_model = args.d_model
num_layer = args.num_layer
num_heads = args.num_heads
theta = args.theta

temperature = args.temperature
top_num = args.top_num

d_ff = (d_model * 8 // 3 // 64) * 64 # must modify

# create Tokenizer instance

tokenizer_data = torch.load(f"results/tokenizer/{tokenizer_input_path}")
merges = tokenizer_data["merges"]
vocab = tokenizer_data["vocab"]
special_tokens = tokenizer_data["special_tokens"]

tokenizer = Tokenizer(vocab, merges, special_tokens)


# create Transformer instance

transLM = TransformerLM(vocab_size, context_len, d_model, num_layer, num_heads, d_ff, theta).to(device)

loadCheckPoint(f"results/checkpoints/{model_path}", transLM)


with open(f"results/prompts/{prefix_path}", "r") as f:
    text = f.read()

encoded_text = tokenizer.encode(text)

def softmax_with_temperature(x: torch.Tensor, temperature: float) -> torch.Tensor:
    x /= temperature
    x_max = torch.max(x, dim=-1, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)

iter = 0


with torch.no_grad():
    while iter < max_length:
        input_ids = torch.tensor(
            encoded_text[-context_len:],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        trans_res = transLM(input_ids)[0, -1, :]

        top_vals, top_idx = torch.topk(trans_res, k=top_num, dim=-1)
        softmaxed = softmax_with_temperature(top_vals, temperature)

        sampled_pos = torch.multinomial(softmaxed, num_samples=1)
        next_token = top_idx[sampled_pos].item()

        encoded_text.append(int(next_token))
        iter += 1

output_text = tokenizer.decode(encoded_text)

with open(f"results/generated_text/{output_path}", "w", encoding="utf-8") as f:
    f.write(output_text)
