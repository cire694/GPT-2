from config import ModelConfig
from model import GPT
import torch

config = ModelConfig(embd_dim=128, n_heads=4, activation='gelu', n_blocks=2,
                    max_seq_len=64, vocab_size=100)
model = GPT(config)
x = torch.randint(0, 100, (2, 16))
y = torch.randint(0, 100, (2, 16))

logits, loss = model(x, y)
print(logits.shape)    # should be (2, 16, 100)
print(loss.item())     # should be ~log(100) ≈ 4.6 at init