import torch
import time
import os
import torch.nn as nn
from collections.abc import Callable
from torch.utils.data import  DataLoader
from config import TrainConfig


class Trainer():

    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        lr_scheduler:Callable,
        config: TrainConfig, 
    ):

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.get_lr = lr_scheduler
        self.config = config
        self.device = next(model.parameters()).device

        self.step = 0
        

    def train(self, out_folder):
        model = self.model
        model.train()
        
        train_iter = iter(self.train_loader)
        while self.step < self.config.max_steps:
            loss, dt, norm = self.train_step(train_iter)

            #maybe also write to a log file
            print(f"step: {self.step:6d} | loss: {loss:.6f} | time: {dt*1000:.2f}ms | norm: {norm:.4f}")
            if self.step > 0 and (self.step % self.config.save_freq == 0 or self.step == self.config.max_steps - 1):
                val_loss = self.validate()
                
                path = os.path.join(out_folder, f"model_{self.step:05d}.pt")
                os.makedirs(out_folder, exist_ok=True)
                
                self.save_checkpoint(val_loss, path)
                print(f"Validation loss: {val_loss:.6f}")
            
            self.step += 1
            
            
    
    def train_step(self, train_iter):
        model = self.model
        model.train()
        optimizer = self.optimizer
        t0 = time.time()

        #we're limited by GPU memory, so we'll simulate one large batch by doing tiny batches
        optimizer.zero_grad() 
        loss_accum = torch.tensor(0.0, device=self.device) # for performance, keep as tensor until very end
        for _ in range(self.config.grad_accum_steps):
            x, y = next(train_iter)
            x, y =  x.to(self.device), y.to(self.device)

            with torch.autocast(self.device.type, self.config.dtype):
                _, loss = model(x, y)
            loss = loss / self.config.grad_accum_steps # scale the loss
            
            loss_accum += loss.detach() # removes from autograd tree.
            loss.backward() # computes gradients
        
        lr = self.get_lr(self.step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


        t1 = time.time()
        return loss_accum.item(), t1 - t0, norm.item()


    def validate(self):
        model = self.model
        model.eval()
        loss_accum = 0
        with torch.no_grad(), torch.autocast(self.device.type, self.config.dtype):
            for i, (x, y) in enumerate(self.val_loader):
                if i >= self.config.max_val_batches:
                    break
                x, y = x.to(self.device), y.to(self.device)
                _, loss = model(x, y)
                loss_accum += loss.detach()
            
        
        return (loss_accum / self.config.max_val_batches).item()
    

    def save_checkpoint(self, val_loss, path):
        checkpoint = {
            'model': self.model.state_dict(),
            'config': self.config,
            'step': self.step, 
            'val_loss': val_loss,
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path): 
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.config = checkpoint['config']
        self.step = checkpoint['step']

                


            
        

