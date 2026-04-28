import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Block
from config import ModelConfig

LOSS_FN = {'cross_entropy': nn.CrossEntropyLoss}

class GPT(nn.Module): 
    
    def __init__(self, config:ModelConfig): 
        super().__init__()
        self.config = config
        self.loss_fn = LOSS_FN[config.loss_fn]()

        # GPT: wte(x) + wpe(x) -> hidden(x) -> layerNorm(x) -> fc_out(x) 
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.embd_dim), #word token embedding. Lookup table, not MatMul
            "wpe": nn.Embedding(config.max_seq_len, config.embd_dim), 
            "hidden": nn.ModuleList([Block(config) for _ in range(config.n_blocks)]),
            "layer_norm": nn.LayerNorm(config.embd_dim)
        })

        self.fc_out = nn.Linear(config.embd_dim, config.vocab_size, bias=False)
        self.fc_out.weight = self.transformer.wte.weight

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding): 
            std = 0.02 / math.sqrt(2 * self.config.n_blocks) if hasattr(m, "IS_RESIDUAL_PROJ") else 0.02
            nn.init.normal_(m.weight, mean=0, std=std)
            if hasattr(m, "bias") and m.bias is not None: 
                nn.init.constant_(m.bias, 0)

    def forward(self, x, labels=None):
        B, T = x.size()
        pos = torch.arange(T, device=x.device) # (T, )
        pos_emb = self.transformer.wpe(pos) #(T, n_embd)
        tok_emb = self.transformer.wte(x) #(B, T, n_embd)

        x = pos_emb + tok_emb
        for block in self.transformer.hidden:
            x = block(x)

        x = self.transformer.layer_norm(x)
        logits = self.fc_out(x) #(B, T, vocab_size)

        #compute loss
        loss = None
        if labels is not None:
            observed = logits.reshape(-1, self.config.vocab_size) #(B * T, vocab_size)
            expected = labels.reshape(-1) #(B * T, )
            loss = self.loss_fn(observed, expected)
        return logits, loss
    
    def generate(self, x, max_length, num_samples = 1, top_k = None, seed=42):
        # x should be tokenized already
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.repeat(num_samples, 1) #(T,) -> (1, T) -> (num_seq, T)
        
        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
        with torch.no_grad():
            while x.size(1) < max_length:
                x = x if x.size(1) < self.config.max_seq_len else x[:, -self.config.max_seq_len:] #if we're over the context window, grab the last window

                logits, _ = self(x)

                logits = logits[:, -1, :] #(num_seq, vocab_size)
                if top_k is not None: 
                    #zero out logits less than the kth largest value
                    topk_values, _ = torch.topk(logits, top_k, -1) #(num_seq, )
                    thresh = topk_values[:, [-1]] #grabs smallest value per sequence. using [:,[-1]] gives shape (num_seq, 1)
                    logits[logits < thresh] = float('-inf')

                probs = F.softmax(logits, -1) #(num_seq, vocab_size)
                next_col = torch.multinomial(probs, num_samples=1, generator=gen)

                x = torch.cat((x, next_col), dim=1)
        return x

        




        



            
        