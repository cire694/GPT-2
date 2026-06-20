import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from config import ModelConfig, TrainConfig
from model import GPT
from trainer import Trainer
from data import FineWebDataSet
from scheduler import make_lr_scheduler

# Steps: 
# 1. Create a DataLoader from existing dataset. Will need to be customized. 
#   - Recommendation: create a new file just for dataloading
# 2. Define model + model configs
# 3. Define trainer + trainer configs
#   - Also define the lr scheduler 
# 4. Train the model with the trainer. 

def main():

    model_config = ModelConfig(
        embd_dim=384, n_heads = 6, n_blocks = 6, 
        max_seq_len=256, vocab_size=50304, 
        activation='gelu'
    )
    train_config = TrainConfig(
        batch_size=64, microbatch_size=8, 
        save_freq=500, max_steps = 2000, max_val_batches=20, 
        max_lr=6e-4, min_lr = 6e-5, warmup_steps=100
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = GPT(model_config).to(device)

    base_data_path = os.path.join(os.path.dirname(__file__), 'edu_fineweb10B')
    train_dataset = FineWebDataSet(base_data_path, seq_len=model_config.max_seq_len, split='train')
    val_dataset = FineWebDataSet(base_data_path, seq_len=model_config.max_seq_len, split='val')
    train_loader = DataLoader(train_dataset, batch_size=train_config.microbatch_size, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config.microbatch_size, num_workers=0, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr = train_config.max_lr, betas=(0.9, 0.95), eps = 1e-8, weight_decay=0.1)
    lr_scheduler = make_lr_scheduler(train_config)

    trainer = Trainer(
        model=model, optimizer=optimizer, 
        train_loader=train_loader, val_loader=val_loader, 
        lr_scheduler=lr_scheduler, config=train_config
    )

    trainer.train(out_folder='shakespeare/checkpoints')

if __name__ == '__main__':
    main()