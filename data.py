import glob
import os

from torch.utils.data import IterableDataset
import numpy as np
import torch
import tiktoken

class ShakespeareDataSet(IterableDataset): 

    def __init__(self, file_path, seq_len): 
        self.seq_len = seq_len
        with open(file_path, 'r') as f: 
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype = torch.long)

    
    def __iter__(self): 
        seq_len = self.seq_len
        n = len(self.tokens)
        pos = 0
        while True: 
            if pos + seq_len + 1 > n:
                pos = 0
            x = self.tokens[pos : pos + seq_len]
            y = self.tokens[pos + 1 : pos + seq_len + 1]
            yield x, y
            pos += seq_len

class FineWebDataSet(IterableDataset): 

    def __init__(self, data_path, seq_len, split='train'):
        self.seq_len = seq_len
        self.data_path = data_path
        self.split = split
        self.shard_paths = self._find_shards(data_path, split)
        if not self.shard_paths:
            raise ValueError(f"No FineWeb shards found for split={split!r} in '{data_path}'")

    def _find_shards(self, data_path, split):
        if os.path.isfile(data_path):
            return [data_path]

        if os.path.isdir(data_path):
            pattern = os.path.join(data_path, f"edufineweb_{split}_*.npy")
            shards = sorted(glob.glob(pattern))
            if shards:
                return shards
            return sorted(glob.glob(os.path.join(data_path, "*.npy")))

        raise ValueError(f"FineWeb data path must be a file or directory, got '{data_path}'")

    def _load_shard(self, index):
        tokens_np = np.load(self.shard_paths[index])
        return torch.from_numpy(tokens_np).long()

    def __iter__(self):
        seq_len = self.seq_len
        shard_idx = 0
        tokens = self._load_shard(shard_idx)
        n = len(tokens)
        pos = 0

        while True:
            if pos + seq_len + 1 > n:
                shard_idx = (shard_idx + 1) % len(self.shard_paths)
                tokens = self._load_shard(shard_idx)
                n = len(tokens)
                pos = 0

            x = tokens[pos : pos + seq_len]
            y = tokens[pos + 1 : pos + seq_len + 1]
            yield x, y
            pos += seq_len

