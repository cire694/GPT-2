from dataclasses import dataclass
from _collections_abc import Callable

@dataclass
class ModelConfig():
    """
    Creates a model configuration class. 
    n_heads: number of heads
    embd_dim: representation size
    num_blocks: number of transformer blocks
    """
    embd_dim: int
    
    #attention
    n_heads: int
    
    #transformer
    num_blocks: int
    activation: function