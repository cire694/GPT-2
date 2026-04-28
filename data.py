from torch.utils.data import IterableDataset
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
    
