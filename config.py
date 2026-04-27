from dataclasses import dataclass
import torch

@dataclass
class ModelConfig():
    """
    Creates a model configuration class.
    embd_dim: dimension of embedding

    n_heads: number of heads
    activation: what type of activation function

    embd_dim: representation size
    n_blocks: number of transformer blocks
    vocab_size: number of unique tokens there are
    """
    embd_dim: int

    #attention
    n_heads: int
    activation: str
    
    #transformer
    n_blocks: int
    max_seq_len: int
    vocab_size: int
    dropout:float = 0.0
    loss_fn:str = 'cross_entropy'
    tokenizer:str = "gpt2"