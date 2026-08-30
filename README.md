# GPT-2 From Scratch in PyTorch

A compact, educational implementation of a GPT-style decoder-only Transformer built from first principles in PyTorch.

This project starts from tokenization and embeddings, develops self-attention step by step, implements layer normalization, GELU feed-forward networks and residual connections, and combines them into a trainable GPT model for autoregressive text generation.

> This is a learning and portfolio project. It reproduces the core architecture and training mechanics of GPT-2-style models; it does **not** provide OpenAI GPT-2 pretrained weights.

## What this project demonstrates

- GPT-2 BPE tokenization with `tiktoken`
- Token and positional embeddings
- Scaled dot-product self-attention
- Causal masking
- Multi-head attention
- Layer normalization
- GELU feed-forward network
- Residual / shortcut connections
- Transformer blocks
- Decoder-only GPT architecture
- Cross-entropy language-model loss
- Train/validation splitting
- Autoregressive text generation
- Temperature and top-k sampling
- Model checkpointing
- Unit tests for important tensor-shape and masking properties

## Architecture

```text
Token IDs
   │
   ├── Token Embeddings
   └── Positional Embeddings
            │
            ▼
       Dropout Layer
            │
            ▼
 ┌─────────────────────────┐
 │ Transformer Block × N   │
 │                         │
 │ LayerNorm               │
 │    ↓                    │
 │ Causal Multi-Head Attn  │
 │    ↓                    │
 │ Residual Connection     │
 │    ↓                    │
 │ LayerNorm               │
 │    ↓                    │
 │ Feed Forward + GELU     │
 │    ↓                    │
 │ Residual Connection     │
 └─────────────────────────┘
            │
            ▼
       Final LayerNorm
            │
            ▼
        Linear Head
            │
            ▼
      Vocabulary Logits
```

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── LICENSE
├── data/
│   └── README.md
├── notebooks/
│   └── README.md
├── src/
│   └── gpt2_scratch/
│       ├── __init__.py
│       ├── config.py
│       ├── model.py
│       ├── data.py
│       ├── generation.py
│       └── training.py
├── scripts/
│   ├── train.py
│   └── generate.py
└── tests/
    └── test_model.py
```

The original learning notebooks can be kept under `notebooks/` in numbered order:

```text
00_tokenization_and_embeddings.ipynb
01_attention_mechanism.ipynb
02_layer_normalization.ipynb
03_feed_forward.ipynb
04_residual_connections.ipynb
05_gpt_architecture_and_training.ipynb
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/GPT_2_from_scratch.git
cd GPT_2_from_scratch

python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

## Quick start

### 1. Train a small GPT model

The default training script uses a smaller configuration than GPT-2 124M so it can be trained on a normal laptop.

Place a UTF-8 text file at:

```text
data/the-verdict.txt
```

Then run:

```bash
python scripts/train.py \
  --data data/the-verdict.txt \
  --epochs 5 \
  --checkpoint checkpoints/model.pt
```

### 2. Generate text

```bash
python scripts/generate.py \
  --checkpoint checkpoints/model.pt \
  --prompt "Every effort moves you" \
  --max-new-tokens 50 \
  --temperature 0.9 \
  --top-k 40
```

## GPT-2 124M configuration

The project also defines the standard GPT-2-small-style architecture:

| Parameter | Value |
|---|---:|
| Vocabulary size | 50,257 |
| Context length | 1,024 |
| Embedding dimension | 768 |
| Attention heads | 12 |
| Transformer layers | 12 |

Training a full 124M-parameter model from scratch requires substantially more data and compute than the included demo corpus.

## Training objective

For a token sequence:

```text
x0, x1, x2, x3, ...
```

the model is trained to predict the next token:

```text
Input:   x0, x1, x2
Target:  x1, x2, x3
```

The loss is token-level cross entropy:

```text
loss = CrossEntropy(logits, target_token_ids)
```

Perplexity can be derived as:

```text
perplexity = exp(loss)
```

## Why causal attention?

GPT is autoregressive: token `t` must not access tokens `t+1`, `t+2`, ...

A triangular causal mask therefore sets future attention scores to negative infinity before softmax:

```text
[ ✓  ✗  ✗  ✗ ]
[ ✓  ✓  ✗  ✗ ]
[ ✓  ✓  ✓  ✗ ]
[ ✓  ✓  ✓  ✓ ]
```

## Tests

Run:

```bash
pytest -q
```

The tests verify:

- output tensor dimensions
- causal masking behavior
- layer normalization
- context-length validation
- generation output shape

## Portfolio notes

This repository is intended to show understanding of the internal mechanics of modern language models rather than simply calling a pretrained model through a library.

Key concepts implemented directly include:

**PyTorch · Transformer Architecture · Self-Attention · Multi-Head Attention · Causal Masking · Layer Normalization · GELU · Residual Connections · Tokenization · Language Modeling · Autoregressive Generation**

## Acknowledgements

The learning progression and small `The Verdict` training corpus are inspired by educational material from Sebastian Raschka's *Build a Large Language Model (From Scratch)* project.

## License

Code in this repository is released under the MIT License. Check the source and redistribution terms of any external datasets separately.
