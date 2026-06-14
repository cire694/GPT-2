# Celltype — Minimal GPT training demo

This repository is a compact, object-oriented implementation of a small GPT-style language model and training loop using PyTorch. It's inspired by Karpathy's "minGPT" / reproduce GPT videos and demonstrates the components you need to train a causal language model on a single text file (the included `input.txt`).

## Key ideas
- Modular, readable PyTorch model code (transformer blocks, attention, MLP).
- Simple data pipeline using `tiktoken` to tokenize text.
- Small `Trainer` class implementing gradient accumulation, validation, checkpointing, and LR scheduling.

## Quick start
1. Create and activate a Python environment (example with conda):

```bash
conda create -n celltype python=3.10 -y
conda activate celltype
pip install torch tiktoken
```

2. Quick smoke test (small forward/backward run):

```bash
python test.py
```

3. Train the model using the example entrypoint:

```bash
python main.py
```

Checkpoints are written to `shakespeare/checkpoints` by default.

## Files & overview
- File: [config.py](config.py) — `ModelConfig` and `TrainConfig` dataclasses (model and training hyperparameters).
- File: [data.py](data.py) — `ShakespeareDataSet`, a simple IterableDataset using `tiktoken` GPT-2 encoding on `input.txt`.
- File: [modules.py](modules.py) — attention, MLP, and transformer `Block` building blocks.
- File: [model.py](model.py) — `GPT` model that composes embeddings, positional encodings, `Block`s and output projection.
- File: [trainer.py](trainer.py) — `Trainer` implementing training loop, gradient accumulation, validation and checkpointing.
- File: [scheduler.py](scheduler.py) — cosine LR schedule with warmup helper `make_lr_scheduler`.
- File: [main.py](main.py) — example training script wiring config, dataloaders, model, optimizer and `Trainer`.
- File: [test.py](test.py) — small smoke test that runs a forward/backward pass on a tiny model.
- File: [input.txt](input.txt) — example training text (used by `ShakespeareDataSet`).

Readme.md generated with the help of Copilot