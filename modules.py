import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


ACTIVATIONS = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}
class MLP(nn.Module): 

    def __init__(self, config:ModelConfig):
        super().__init__()
        self.fc_in = nn.Linear(config.embd_dim, 4 * config.embd_dim)
        self.activation = ACTIVATIONS[config.activation]()
        self.fc_out = nn.Linear(4 * config.embd_dim, config.embd_dim)

        # scale output projection variance by 1/sqrt(2 * n_blocks) of normal variance
        self.fc_out.IS_RESIDUAL_PROJ = True

    
    def forward(self, x: torch.Tensor):
        return self.fc_out(self.activation(self.fc_in(x)))
    
class Attention(nn.Module): 

    def __init__(self, config:ModelConfig, is_causal=False): 
        super().__init__()

        self.is_causal = is_causal
        self.embd_dim = config.embd_dim
        self.n_heads = config.n_heads
        assert self.embd_dim % self.n_heads == 0
        self.head_dim = self.embd_dim // self.n_heads

        self.qkv = nn.Linear(config.embd_dim, 3 * config.embd_dim)
        self.fc_out = nn.Linear(config.embd_dim, config.embd_dim)

        # scale output projection variance by 1/sqrt(2 * n_blocks) of normal variance
        self.fc_out.IS_RESIDUAL_PROJ = True


    def forward(self, x:torch.Tensor): 
        B, T, _ = x.size() 
        
        # (B, T, embd_dim) -> (B, T, 3 * embd_dim), then split it
        q, k, v = self.qkv(x).split(self.embd_dim, dim=2)

        # reshaping qkv: (B, T, embd_dim) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        output = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        output = output.transpose(1, 2).reshape(B, T, -1) #(B, n_heads, T, head_dim) => (B, T, n_heads, head_dim)  => (B, T, embd_dim)
        return self.fc_out(output)

class Block(nn.Module): 

    def __init__(self, config:ModelConfig): 
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.embd_dim)
        self.attention = Attention(config, is_causal=True)
        self.ln_2 = nn.LayerNorm(config.embd_dim)
        self.mlp = MLP(config)

    def forward(self, x:torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
    

        




        





