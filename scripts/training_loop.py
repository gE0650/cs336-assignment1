
import argparse, torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from cs336_basics.base_modules import TransformerLM
from cs336_basics.training_modules import loadBatch, AdamW, crossEntrophy, gradClip, cosLRS, saveCheckPoint, loadCheckPoint

parser = argparse.ArgumentParser(description="Training loop")
parser.add_argument("--input", required=True, help="tokenized input")
parser.add_argument("--output", required=True, help="output model data")
parser.add_argument("--resume-from", required=False, help="resume from model data")
parser.add_argument("--log-dir", required=True, help="name of log dir")


parser.add_argument("--batch-size", required=True, help="batch size", type=int)
parser.add_argument("--iter-num", required=True, help="iteration number", type=int)

parser.add_argument("--vocab-size", required=True, help="vocabulary size", type=int)
parser.add_argument("--context-len", required=True, help="context length", type=int)
parser.add_argument("--d-model", required=True, help="d_model", type=int)
parser.add_argument("--num-layer", required=True, help="num_layer", type=int)
parser.add_argument("--num-heads", required=True, help="num_head", type=int)
parser.add_argument("--theta", required=True, help="rope theta", type=float)


parser.add_argument("--weight-decay", required=True, help="weight decay", type=float)
parser.add_argument("--b1", required=True, help="beta1", type=float)
parser.add_argument("--b2", required=True, help="beta2", type=float)
parser.add_argument("--max-grad", required=True, help="max gradient mean", type=float)

parser.add_argument("--a-max", required=True, help="a_max", type=float)
parser.add_argument("--a-min", required=True, help="a_min", type=float)
parser.add_argument("--T-w", required=True, help="T_w", type=int)
parser.add_argument("--T-c", required=True, help="T_c", type=int)

parser.add_argument("--device", required=True, help="device")

args = parser.parse_args()

input_path = args.input
output_path = args.output
resume_path = args.resume_from
log_dir = args.log_dir

batch_size = args.batch_size
iter_num = args.iter_num


device = args.device
vocab_size = args.vocab_size
context_len = args.context_len
d_model = args.d_model
num_layer = args.num_layer
num_heads = args.num_heads
theta = args.theta


weight_decay = args.weight_decay
betas = (args.b1, args.b2)
max_grad = args.max_grad

a_max = args.a_max
a_min = args.a_min
T_w = args.T_w
T_c = args.T_c

d_ff = (d_model * 8 // 3 // 64) * 64 # must modify

config = (vocab_size, context_len, d_model, num_layer, num_heads, d_ff, theta)

tokenized_data = np.memmap(f"results/tokenized_text/{input_path}", dtype=np.uint16, mode="r")

transLM = TransformerLM(vocab_size, context_len, d_model, num_layer, num_heads, d_ff, theta).to(device)
optimizer = AdamW(transLM.parameters(), 0, weight_decay, betas, 1e-8)

writer = SummaryWriter(log_dir=f"results/tensorboard/{log_dir}")

iter = 0

if resume_path is not None:
    iter = loadCheckPoint(f"results/checkpoints/{resume_path}", transLM, optimizer) + 1



while iter < iter_num:
    (input_tokens, target) = loadBatch(tokenized_data, batch_size, context_len, device)
    result = transLM(input_tokens)
    loss = crossEntrophy(result, target)
    
    loss.backward()

    gradClip(transLM.parameters(), max_grad)

    for group in optimizer.param_groups:
        group["lr"] = cosLRS(iter, a_max, a_min, T_w, T_c)
    optimizer.step()

    if iter % 5 == 0:
        if iter % 50 == 0:
            saveCheckPoint(transLM, optimizer, iter, f"results/checkpoints/{output_path}_{iter + 1}")

        with torch.no_grad():
            (valid_input_tokens, valid_target) = loadBatch(tokenized_data, batch_size, context_len, device)
            valid_result = transLM(valid_input_tokens)
            valid_loss = crossEntrophy(valid_result, valid_target)

        loss_dict = {
            "train": loss.detach().item(),
            "valid": valid_loss.detach().item(),
        }
        writer.add_scalars("loss", loss_dict, iter + 1)

    transLM.zero_grad()
    iter += 1

writer.close()