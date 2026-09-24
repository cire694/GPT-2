# GPT-2-style transformer from scratch

A from-scratch PyTorch implementation of a GPT-2-style decoder-only transformer,
trained on FineWeb-Edu. Built while following Andrej Karpathy's GPT-2 reproduction
series, with my own modular structure (separate model, trainer, scheduler, and config).

## Status

Trained for 46K steps (~760M tokens) of a planned 600K-step run. The cosine LR
schedule was configured for the full 600K steps, so the learning rate was still
near its peak when training stopped. A full run with the schedule matched to the
training length is in progress; results and loss curves will be added here.

## Model

- ~30M parameters (~10.6M non-embedding + ~19.3M tied embedding)
- 6 transformer blocks, 6 attention heads, 384-dim embeddings, 256-token context
- Multi-head causal self-attention, GELU MLP, pre-LayerNorm blocks
- Learned token and positional embeddings
- Weight tying between the token embedding and output projection
- Residual projections initialized with std scaled by 1/sqrt(2 * n_layers)

## Training

- Data: FineWeb-Edu 10B-token sample, GPT-2 tokenizer via `tiktoken`
- Batch: 64 sequences × 256 tokens per step (8 micro-batches of 8 with gradient accumulation)
- AdamW (β = 0.9, 0.95; weight decay 0.1), gradient clipping at 1.0
- LR: 5,000-step linear warmup, then cosine decay from 6e-4 to 6e-5
- bf16 autocast
- Validation every 500 steps on 20 held-out batches

## Run it

```bash
pip install torch tiktoken numpy
python fineweb.py                      # download and tokenize FineWeb-Edu into shards
python train_fineweb.py | tee train.log
```

## Files

- `model.py`: GPT model (embeddings, blocks, output head, generation)
- `modules.py`: attention, MLP, and transformer block
- `trainer.py`: training loop, gradient accumulation, validation, checkpointing
- `scheduler.py`: warmup + cosine LR schedule
- `config.py`: model and training configs
- `data.py`: FineWeb-Edu and Shakespeare datasets
- `train_fineweb.py`: FineWeb-Edu training run
- `train_shakespeare.py`: small sanity-check run on Tiny Shakespeare
