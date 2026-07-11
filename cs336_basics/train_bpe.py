import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    #assert isinstance(special_tokens, list[bytes]), "Must represent special token as a bytestring"
    

    chunk_boundaries = [0]

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if not special_tokens:
        return [0, file_size]

    max_special_token_len = max([len(token) for token in special_tokens])

    #special_token = set(special_token)
    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess

        while True:
            file.seek(initial_position)
            mini_chunk = file.read(mini_chunk_size + max_special_token_len)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = [mini_chunk.find(special_token) for special_token in special_tokens]
            found_at = min([ite for ite in found_at if ite != -1], default=-1)

            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size

            

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    chunk_boundaries = sorted(set(chunk_boundaries))
    return chunk_boundaries

def pretoken_from_chunk(
    PAT: str,
    chunk: bytes,
    special_tokens: list[str],
) -> dict[bytes, int]:

    text = chunk.decode("utf-8")
    
    # remove special tokens
    if special_tokens:
        escaped_tokens = [re.escape(token) for token in special_tokens]
        separator_pattern = "|".join(escaped_tokens)
        text_parts = re.split(separator_pattern, text)
    else:
        text_parts = [text]

    # count pre_token num
    res = {}
    for text_part in text_parts:
        for match in re.finditer(PAT, text_part):
            pre_token_bytes = match.group(0).encode("utf-8")
            res[pre_token_bytes] = res.get(pre_token_bytes, 0) + 1

    return res




def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # chunknization
    
    chunks = []
    special_tokens_bytes = [token.encode("utf-8") for token in special_tokens]

    cpu_num = os.cpu_count() or 1

    with open(input_path, "rb") as f:
        # chunknize
        chunk_boundaries = find_chunk_boundaries(f, cpu_num, special_tokens_bytes)
        
        f.seek(0)
        for i in range(1, len(chunk_boundaries)):
            f.seek(chunk_boundaries[i - 1])
            chunks.append(f.read(chunk_boundaries[i] - chunk_boundaries[i - 1]))

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # count num of pre_tokens
    
    num_of_processes = len(chunk_boundaries) - 1
    pre_tokens = {}

    tasks = [(PAT, chunk, special_tokens) for chunk in chunks]
    with Pool(processes=num_of_processes) as pool:
        results_pre_token = pool.starmap(pretoken_from_chunk, tasks)

    for result_pre_token in results_pre_token:
        for pre_token, num in result_pre_token.items():
            pre_tokens[pre_token] = pre_tokens.get(pre_token, 0) + num
        
    # initialize vocab
    vocab = {}
    merges = []

    current_vocab_size = 256
    for i in range(256):
        vocab[i] = bytes([i])
    for i in range(len(special_tokens_bytes)):
        vocab[256 + i] = special_tokens_bytes[i]
        current_vocab_size += 1

    # initial pairs_count
    pairs_count = {}
    all_tokens = []

    for (pre_token, num_of_pre_token) in pre_tokens.items():
        all_tokens.append(([bytes([b]) for b in pre_token], num_of_pre_token))
        

    while current_vocab_size < vocab_size:

        pairs_count = {}
        
        for (tokens, num_of_pre_token) in all_tokens:
            token_pair = list(zip(tokens, tokens[1:]))
            for pair in token_pair:
                pairs_count[pair] = pairs_count.get(pair, 0) + num_of_pre_token
        
        if not pairs_count:
            break

        best_pair, best_count = max(pairs_count.items(), key=lambda item: (item[1], item[0]))

        vocab[current_vocab_size] = best_pair[0] + best_pair[1]
        current_vocab_size += 1
        merges.append(best_pair)

        new_all_tokens = []

        for tokens, num_of_pre_token in all_tokens:
            new_tokens = []

            i = 0
            while i < len(tokens):
                if (
                    i + 1 < len(tokens)
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            new_all_tokens.append((new_tokens, num_of_pre_token))

        all_tokens = new_all_tokens

        
    
    return vocab, merges