"""
Streaming Data Pipelines for superGPT
========================================
Production-grade data loading for multi-terabyte datasets.
Supports sharded binary files, HuggingFace datasets streaming,
and on-the-fly tokenization.

Usage:
    # Shard a large tokenized dataset
    python streaming.py shard --input data/train.bin --n-shards 64 --output data/shards/

    # Use sharded data in training
    python train.py --preset large --data-shards data/shards/

    # Stream from HuggingFace
    python train.py --preset medium --hf-dataset HuggingFaceFW/fineweb --streaming

    # Tokenize and stream from raw text files
    python train.py --preset medium --text-glob 'corpus/*.txt' --tokenizer tiktoken

Reference:
    Webdataset, Mosaic StreamingDataset, HuggingFace datasets
"""

import os
import sys
import glob
import math
import pickle
import random
import argparse
from typing import Optional, Iterator, List

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader


# ==============================================================================
#  Sharded Binary Dataset
# ==============================================================================

class ShardedBinDataset(IterableDataset):
    """Streams data from multiple pre-tokenized .bin shard files.

    Each shard is a flat array of token IDs. Shards are shuffled per epoch,
    then sampled randomly from each shard.

    Args:
        shard_dir: directory containing shard_000.bin, shard_001.bin, ...
        block_size: sequence length
        shuffle: whether to shuffle shard order each epoch
        world_size: number of data-parallel workers (for sharding)
        rank: this worker's rank
    """

    def __init__(self, shard_dir: str, block_size: int, shuffle: bool = True,
                 world_size: int = 1, rank: int = 0, dtype=np.uint16):
        super().__init__()
        self.block_size = block_size
        self.shuffle = shuffle
        self.dtype = dtype

        # Find all shard files
        self.shard_paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.bin")))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard files found in {shard_dir}")

        # Assign shards to this rank (round-robin)
        self.shard_paths = self.shard_paths[rank::world_size]

        # Get total tokens for progress reporting
        self.total_tokens = 0
        for path in self.shard_paths:
            size = os.path.getsize(path)
            self.total_tokens += size // np.dtype(dtype).itemsize

        print(f"  [Rank {rank}] {len(self.shard_paths)} shards, "
              f"{self.total_tokens/1e6:.1f}M tokens")

    def __iter__(self) -> Iterator:
        worker_info = torch.utils.data.get_worker_info()
        shard_list = list(self.shard_paths)

        # Sub-divide shards among DataLoader workers
        if worker_info is not None:
            n_workers = worker_info.num_workers
            worker_id = worker_info.id
            shard_list = shard_list[worker_id::n_workers]

        if self.shuffle:
            random.shuffle(shard_list)

        for shard_path in shard_list:
            data = np.memmap(shard_path, dtype=self.dtype, mode='r')
            n_tokens = len(data)

            if n_tokens <= self.block_size:
                continue

            # Random sampling from this shard
            n_samples = n_tokens // self.block_size
            indices = list(range(n_tokens - self.block_size))
            if self.shuffle:
                random.shuffle(indices)

            for idx in indices[:n_samples]:
                x = torch.from_numpy(data[idx:idx + self.block_size].astype(np.int64))
                y = torch.from_numpy(data[idx + 1:idx + 1 + self.block_size].astype(np.int64))
                yield x, y


# ==============================================================================
#  HuggingFace Streaming Dataset
# ==============================================================================

class HFStreamingDataset(IterableDataset):
    """Stream from HuggingFace datasets without downloading entire dataset.

    Args:
        dataset_name: HuggingFace dataset name (e.g. 'HuggingFaceFW/fineweb')
        tokenizer_name: tokenizer to use ('tiktoken' or HF tokenizer name)
        block_size: sequence length
        split: dataset split
    """

    def __init__(self, dataset_name: str, tokenizer_name: str = "tiktoken",
                 block_size: int = 256, split: str = "train",
                 text_column: str = "text"):
        super().__init__()
        self.block_size = block_size
        self.text_column = text_column

        # Load tokenizer
        if tokenizer_name == "tiktoken":
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            self.encode = enc.encode
            self.vocab_size = enc.n_vocab
        else:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            self.encode = lambda t: tok.encode(t, add_special_tokens=False)
            self.vocab_size = tok.vocab_size

        # Load streaming dataset
        try:
            from datasets import load_dataset
            self.dataset = load_dataset(
                dataset_name, split=split, streaming=True,
                trust_remote_code=True,
            )
        except ImportError:
            print("Error: HuggingFace streaming requires 'datasets' package.")
            print("Install with: pip install datasets")
            sys.exit(1)

        print(f"  Streaming: {dataset_name} ({split})")
        print(f"  Tokenizer: {tokenizer_name} (vocab={self.vocab_size})")

    def __iter__(self) -> Iterator:
        buffer = []

        for example in self.dataset:
            text = example.get(self.text_column, "")
            if not text:
                continue

            tokens = self.encode(text)
            buffer.extend(tokens)

            # Yield complete blocks from buffer
            while len(buffer) >= self.block_size + 1:
                chunk = buffer[:self.block_size + 1]
                buffer = buffer[self.block_size:]

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


# ==============================================================================
#  Text Glob Dataset (on-the-fly tokenization)
# ==============================================================================

class TextGlobDataset(IterableDataset):
    """Stream from raw text files with on-the-fly tokenization.

    Args:
        pattern: glob pattern for text files (e.g. 'corpus/*.txt')
        tokenizer_name: tokenizer to use
        block_size: sequence length
    """

    def __init__(self, pattern: str, tokenizer_name: str = "tiktoken",
                 block_size: int = 256):
        super().__init__()
        self.block_size = block_size
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files matching: {pattern}")

        # Load tokenizer
        if tokenizer_name == "tiktoken":
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            self.encode = enc.encode
        else:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            self.encode = lambda t: tok.encode(t, add_special_tokens=False)

        print(f"  Text files: {len(self.files)} files matching '{pattern}'")

    def __iter__(self) -> Iterator:
        files = list(self.files)
        random.shuffle(files)
        buffer = []

        for fpath in files:
            with open(fpath, 'r', errors='ignore') as f:
                for line in f:
                    tokens = self.encode(line)
                    buffer.extend(tokens)

                    while len(buffer) >= self.block_size + 1:
                        chunk = buffer[:self.block_size + 1]
                        buffer = buffer[self.block_size:]
                        x = torch.tensor(chunk[:-1], dtype=torch.long)
                        y = torch.tensor(chunk[1:], dtype=torch.long)
                        yield x, y


# ==============================================================================
#  Sharding Utility
# ==============================================================================

def shard_dataset(input_path: str, output_dir: str, n_shards: int, dtype=np.uint16):
    """Split a large .bin file into N shards for distributed training.

    Args:
        input_path: path to input .bin file
        output_dir: directory for output shard files
        n_shards: number of shards to create
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read input
    data = np.memmap(input_path, dtype=dtype, mode='r')
    total_tokens = len(data)
    tokens_per_shard = math.ceil(total_tokens / n_shards)

    print(f"Sharding {input_path}: {total_tokens} tokens -> {n_shards} shards")

    for i in range(n_shards):
        start = i * tokens_per_shard
        end = min(start + tokens_per_shard, total_tokens)
        shard = np.array(data[start:end], dtype=dtype)

        shard_path = os.path.join(output_dir, f"shard_{i:04d}.bin")
        shard.tofile(shard_path)
        print(f"  shard_{i:04d}.bin: {len(shard)} tokens")

    print(f"Done. {n_shards} shards in {output_dir}")


# ==============================================================================
#  Factory function for train.py integration
# ==============================================================================

def create_streaming_dataloader(args, block_size: int, batch_size: int,
                                world_size: int = 1, rank: int = 0):
    """Create a streaming DataLoader based on args.

    Called from train.py when streaming options are specified.
    Returns a DataLoader that yields (x, y) batches.
    """
    if hasattr(args, 'data_shards') and args.data_shards:
        # Sharded binary files
        meta_path = os.path.join(os.path.dirname(args.data_shards), "meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            dtype = np.uint32 if meta.get("tokenizer_type") == "tiktoken" else np.uint16
        else:
            dtype = np.uint16

        dataset = ShardedBinDataset(
            args.data_shards, block_size, shuffle=True,
            world_size=world_size, rank=rank, dtype=dtype,
        )
    elif hasattr(args, 'hf_dataset') and args.hf_dataset:
        # HuggingFace streaming
        tokenizer = getattr(args, 'tokenizer', 'tiktoken')
        dataset = HFStreamingDataset(
            args.hf_dataset, tokenizer, block_size,
        )
    elif hasattr(args, 'text_glob') and args.text_glob:
        # Raw text files
        tokenizer = getattr(args, 'tokenizer', 'tiktoken')
        dataset = TextGlobDataset(
            args.text_glob, tokenizer, block_size,
        )
    else:
        return None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
        drop_last=True,
    )


def get_streaming_args(parser):
    """Add streaming data arguments to an argument parser."""
    parser.add_argument("--data-shards", type=str, default=None,
                        help="Directory with shard_*.bin files")
    parser.add_argument("--hf-dataset", type=str, default=None,
                        help="HuggingFace dataset name for streaming")
    parser.add_argument("--text-glob", type=str, default=None,
                        help="Glob pattern for raw text files")
    parser.add_argument("--tokenizer", type=str, default="tiktoken",
                        help="Tokenizer for streaming: tiktoken or HF model name")
    return parser


# ==============================================================================
#  CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming data utilities for superGPT")
    sub = parser.add_subparsers(dest="command")

    # Shard command
    shard_cmd = sub.add_parser("shard", help="Split a .bin file into N shards")
    shard_cmd.add_argument("--input", required=True, help="Input .bin file")
    shard_cmd.add_argument("--output", required=True, help="Output shard directory")
    shard_cmd.add_argument("--n-shards", type=int, default=64, help="Number of shards")

    args = parser.parse_args()

    if args.command == "shard":
        shard_dataset(args.input, args.output, args.n_shards)
    else:
        parser.print_help()
