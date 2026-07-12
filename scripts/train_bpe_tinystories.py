from cs336_basics.train_bpe import train_bpe

input_path = "data/TinyStoriesV2-GPT4-train.txt"
output_path = "results/train_bpe_tinystories_train.txt"
vocab_size = 10000
special_tokens = ["<|endoftext|>"]

vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

with open(output_path, "w", encoding="utf-8") as f:
    for merge in merges:
        f.write(f"{merge[0]!r}\t{merge[1]!r}\t->\t{(merge[0] + merge[1])!r}\n")

