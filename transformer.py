import torch
import torch.nn as nn
from collections.abc import Callable
from config import ModelConfig


class MLP(): 

    def __init__(self, config:ModelConfig):
        self.fc = nn.Linear(in_features=config.embd_dim, out_features=config.embd_dim * 4)
        self.activation:Callable = config.activation
