from dataclasses import dataclass
from collections.abc import Callable
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

@dataclass
class TrainConfig(): 
    batch_size: int
    microbatch_size: int #how many sequences in one forward pass
    

    save_freq: int
    max_steps:int
    max_val_batches: int
    max_lr: float = 6e-4
    min_lr: float = 6e-5

    warmup_steps: int = 0
    dtype: torch.dtype = torch.bfloat16


    def __post_init__(self):
        assert self.batch_size % self.microbatch_size == 0
        self.grad_accum_steps = self.batch_size // self.microbatch_size #how many forward + backward passes before optimizer.step()

    