# Project Overview

MeshTron is a deep learning model for generating 2D quadrilateral mesh topologies autoregressively. Given a point cloud and a target face count, it generates a sequence of tokens representing quad mesh faces.

## Running the Code

There is no package build system. Run scripts directly with Python.

```bash
# Training
python main.py          # main training entry point (hardcoded config for centered_blades.pt dataset)
python training.py      # alternate entry point (work in progress)

# Evaluation
python testing.py       # load checkpoint and run inference/generation
python validation.py    # inspect checkpoints and plot training history
```

Dependencies (not in requirements.txt — install manually): `torch`, `torch_geometric`, `openmesh`, `numpy`, `matplotlib`, `tqdm`.

## Architecture

The core model (`meshtron.py`) combines:
- **PerceiverPointEncoder** (`point_encoder.py`): Cross-attention over 2000 sampled boundary+interior points → latent geometry embeddings
- **FaceCountEncoder** (`faceCount_encoder.py`): Sinusoidal encoding of the target face count
- **HourglassTransformer** (`hourglass_transformer.py`): 5-stage bottleneck transformer that processes the token sequence at multiple resolutions

### HourglassTransformer stages
1. Coordinate level (raw token resolution)
2. Vertex level (2× shortening — 4 tokens → 1 per vertex)
3. Face level (4× shortening — 8 tokens → 1 per face)
4. Vertex level reconstruction (4× upsampling + residual)
5. Coordinate level reconstruction (2× upsampling + residual)

Each stage applies causal self-attention plus cross-attention conditioning on the geometry and face-count embeddings. Shortening/upsampling layers are in `hourglass_transformer.py`.

### Tokenization (`tokenizer_v2.py`)
- 8 tokens per quad face (4 vertices × 2 coordinates), each quantized into discrete levels (default: 1024)
- Vocabulary size = `quantization_levels + 3` (+ BOS, EOS, PAD special tokens)
- Four sorting strategies controlled by `sorting_strategy` flag: `0` = lexicographical, `1` = topological DFS, `2` = layered x-preferred, `3` = none

### Training (`trainer.py`, `dataset.py`)
- `MeshData` dataset creates `(input_tokens, target_tokens)` pairs shifted by 1 for autoregressive next-token prediction
- Loss: cross-entropy ignoring PAD tokens
- Optimizer: AdamW + cosine annealing LR scheduler
- Supports gradient accumulation and sliding window training for long sequences
- Checkpoints saved to directories named: `q_<quant>_d_model_<d>_n_latents_<nl>_batch_size_<bs>_n_heads_<nh>_window_size_<ws>_sorting_strategy_<ss>_stage_layers_<l1>_<l2>_<l3>_<l4>_<l5>/`

### Attention (`attention.py`, `positional_encoder.py`)
- Multi-head attention with Rotary Position Embeddings (RoPE)
- `is_causal` flag controls causal masking (used for autoregressive self-attention stages; cross-attention uses `is_causal=False`)
- Position IDs are tracked explicitly to support windowed training

## Key Conventions

- **Config as code**: hyperparameters (quantization, d_model, n_latents, batch_size, n_heads, window_size, stage_layers, sorting_strategy) are set directly in `main.py` or `testing.py` — there are no config files.
- **`tokenizer_v2.py` is the current tokenizer**; `tokenizer.py` is the deprecated v1.
- **`training.py` is incomplete**; `main.py` is the working entry point.
