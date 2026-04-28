import math
from config import TrainConfig

def make_lr_scheduler(config:TrainConfig): 
    def get_lr(step): 
        if step < config.warmup_steps:
            return config.max_lr * (step + 1) / config.warmup_steps
        if step >= config.max_steps:
            return config.min_lr
        
        decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.min_lr + coeff * (config.max_lr - config.min_lr)
    return get_lr