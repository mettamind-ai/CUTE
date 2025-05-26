import dataclasses
import glob
import json
import logging
import sys
import time
from pathlib import Path
from tqdm import tqdm

import bpeasy
from bpeasy.tokenizer import BPEasyTokenizer

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@dataclasses.dataclass
class TrainBPETokenizerArgs:
    dataset: str = "./"
    max_sentencepiece_length: int = 128
    regex_pattern: str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

    def __post_init__(self):
        checkpoint_dir = Path(self.dataset)
        assert checkpoint_dir.is_dir(), checkpoint_dir

import lzma
def jsonl_content_iterator(
    args: TrainBPETokenizerArgs,
):
    """
    Iterates over a jsonl file and yields the content of each line
    Tracks the number of characters yielded and stops when the limit is reached
    This is ripe for optimisation if you want to mess with more fine-grained
    character limits (eg. more Python than Java)
    """
    file_path = args.dataset
    chunk_num, character_count = 0, 0
    chunks = glob.glob(f"{file_path}/*.jsonl.xz")

    while chunk_num < len(chunks):
        file_name = chunks[chunk_num]
        print(file_name)
        with lzma.open(file_name, "r") as f:
            for line in f:
                obj = json.loads(line)
                text = obj["text"]
                text_character_count = len(text)
                character_count += text_character_count
                yield text
        chunk_num += 1


def train_bpeasy(args: TrainBPETokenizerArgs):
    return BPEasyTokenizer.train(
        jsonl_content_iterator(args),
        vocab_size=args.vocab_size - len(args.special_tokens),
        regex_pattern=args.regex_pattern,
        special_tokens=args.special_tokens,
        fill_to_nearest_multiple_of_eight=True,
        name=f"bpeasy{args.vocab_size}",
    )


def encode(tokenizer, args) -> float:
    iterator = jsonl_content_iterator(args)
    lengths = []
    num_bytes = 0
    for text in iterator:
        num_bytes += len(text.encode("utf-8"))
        encoded = tokenizer.encode(text)
        lengths.append(len(encoded))
    return num_bytes / sum(lengths)


def get_mean_std_dev(times: list[float]) -> tuple[float, float]:
    avg_time = sum(times) / len(times)
    std_dev = sum([(t - avg_time) ** 2 for t in times])
    return avg_time, std_dev

special_tokens = ["<|begin-of-text|>", "<|end-of-text|>"]
if __name__ == "__main__":
    args = TrainBPETokenizerArgs()
    args.vocab_size = 6400
    args.special_tokens = special_tokens
    tokenizer = train_bpeasy(args)
    tokenizer.save(f"bpeasy{args.vocab_size}.json")
