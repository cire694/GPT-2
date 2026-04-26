import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable
from config import ModelConfig



class MLP(nn.Module): 

    def __init__(self, config:ModelConfig):
        self.fc_in = nn.Linear(config.embd_dim, 4 * config.embd_dim)
        self.activation: Callable = config.activation
        self.fc_out = nn.Linear(4 * config.embd_dim, config.embd_dim)
    
    def forward(self, x: torch.Tensor):
        return self.fc_out(self.activation(self.fc_in(x))) #can we use dot notation?
    
class Attention(nn.Module): 

    def __init__(self, config:ModelConfig, is_causal=False): 
        self.is_causal = is_causal
        self.embd_dim = config.embd_dim
        self.n_heads = config.n_heads
        assert(self.embd_dim % self.n_heads == 0)
        self.head_dim = self.embd_dim // self.n_heads

        self.qkv = nn.Linear(config.embd_dim, 3 * config.embd_dim)

    def forward(self, x:torch.Tensor): 
        B, T, _ = x.size() #should we assert that it matches expected dimensions? e.g. C = embd_dim
        
        # (B, T, embd_dim) -> (B, T, 3 * embd_dim), then split it
        q, k, v = self.qkv(x).split(self.embd_dim, dim=2)

        # reshaping qkv: (B, T, embd_dim) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        return F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)

class Block(nn.Module): 

    def __init__(self, config:ModelConfig): 
        
        self.ln_in = nn.LayerNorm(config.embd_dim)
        self.attention = Attention(config)
        self.ln_out = nn.LayerNorm(config.embd_dim) #why can't we use the same layer norm?
        self.mlp = MLP(config)

    def forward(self, x:torch.Tensor):
        x = x + self.attention(self.ln_in(x))
        x = x + self.mlp(self.ln_out(x))
        return x
    
    

        




        





