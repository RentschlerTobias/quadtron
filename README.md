# Meshtron

Autoregressive transformer for 2D quadrilateral mesh generation. Conditioned on a point cloud and a target face count, it generates a token sequence describing quad faces.

## Running

No package build. Run scripts directly:

```bash
python train.py                 # train with TrainingConfig defaults
python train.py --config x.json # train from a JSON config (CLI flags override fields)
python testing.py               # load checkpoint and run inference / generation
python validation.py            # inspect checkpoints and plot training history
```

Dependencies (install manually): `torch`, `torch_geometric`, `openmesh`, `numpy`, `matplotlib`, `tqdm`.

## Module layout

### Training pipeline
- `config.py` — `TrainingConfig` dataclass: single source of truth for all hyperparameters; hashable, JSON-serializable.
- `reproducibility.py` — `set_seed`, DataLoader generator and worker init for deterministic runs.
- `metrics.py` — `TokenLossAccumulator` and `EpochMetrics`. Computes NLL, bits-per-token and perplexity weighted by valid (non-pad) tokens, so values are comparable across batch sizes and sequence lengths.
- `policy.py` — `Policy` wrapper around `Meshtron`. `logits()` for teacher forcing today, `sample()` is in place for a future RL phase.
- `objectives.py` — `Objective` ABC + `TeacherForcingObjective` (cross-entropy, sum-reduction, pad-ignored). Returns `(loss, loss_sum, n_tokens)` so the per-step gradient is on per-token scale while logging stays unbiased.
- `logger.py` — `JSONLLogger`. Per run: `runs/<config-hash>/{config.json, metrics.jsonl, result.json}`.
- `trainer.py` — `Trainer(cfg).run() -> RunResult`. Linear warmup + cosine schedule, bf16/fp16 autocast, correct gradient accumulation, opt-in checkpointing.
- `train.py` — CLI entry point. Builds `TrainingConfig` from defaults / JSON / flags and calls `Trainer.run()`.

### Model
- `meshtron.py` — `Meshtron` model: token embedding + point encoder + face-count encoder, fed into the transformer with causal self-attention and cross-attention to the latent condition.
- `hourglass_transformer.py` — currently a flat transformer (despite the historical name): each stage runs at full sequence length, followed by cross-attention conditioning. No shortening / upsampling, so no information leaks through downsampling.
- `attention.py`, `positional_encoder.py` — multi-head attention with RoPE; `is_causal` flag controls masking.
- `point_encoder.py` — `PerceiverPointEncoder`: cross-attention over sampled boundary + interior points to a fixed-size latent set, with Fourier features and pre-norm.
- `faceCount_encoder.py` — sinusoidal encoding of the target face count.
- `embedding.py` — token embedding (positional encoding currently disabled there; positions handled inside attention via RoPE).

### Data
- `tokenizer_v2.py` — current tokenizer. 8 tokens per quad face (4 vertices × 2 coordinates), each quantized into discrete levels (default 256). Vocabulary size = `quantization_levels + 3` (BOS, EOS, PAD). `sorting_strategy=1` (adjacent / topological row-direction ordering) is the only supported strategy after the recent cleanup. `tokenizer.py` is the legacy v1.
- `dataset.py` — `MeshData`. Tokenizes meshes, samples a fixed-size point cloud (boundary first, then interior, with noise replication if interior is small), and produces shifted `(input_tokens, target_tokens)` pairs for next-token prediction.

## Loss and metrics

Cross-entropy is computed with `reduction='sum'` and divided by the number of non-pad target tokens. Two consequences:

1. The backward signal is on per-token scale, so the optimal learning rate is largely portable across `batch_size` and `accumulation_steps`.
2. Logged metrics are token-weighted means (`Σ nll_i / Σ tokens`), not means-of-batch-means. They are comparable across runs with different batch sizes, sequence lengths and padding ratios.

Reported per epoch:
- `nll_per_token` — natural-log NLL per valid token.
- `bits_per_token` — same in bits (`nll / ln 2`). Useful upper bound: `log2(vocab_size)` ≈ 8.0 for `quantization=256`.
- `perplexity` — `exp(nll)`.

The tqdm postfix (`bpt=…`) shows the running token-weighted bits-per-token from the start of the current epoch. The outer bar tracks epochs against `cfg.num_epochs` and stops early if no validation improvement for `early_stopping_patience` epochs.

## Run artefacts

```
runs/<config-hash>/
    config.json     # full TrainingConfig snapshot
    metrics.jsonl   # one JSON record per logged epoch
    result.json     # best_val_bpt, best_epoch, total runtime, ...
    best.pt         # only if cfg.save_best
    last.pt         # only if cfg.save_last
```

The config hash is a deterministic 8-char digest of all fields, so identical configs share a directory and accidental duplicates are obvious.

## Conventions

- All hyperparameters live in `TrainingConfig`. Don't hardcode them inside `Trainer`, `Meshtron` or sweeps.
- Don't average per-batch loss values across batches; use `TokenLossAccumulator`.
- New training objectives plug in by implementing `Objective.compute(batch, policy) -> ObjectiveOutput`. The trainer is objective-agnostic.
- Checkpointing is off by default; turn on explicitly per run via `--save-best` / `--save-last`.

## Hardware notes

`precision="bf16"` is the default and works on Ampere (RTX 3090) and newer. Use `"fp16"` only on hardware without bf16 (the `GradScaler` is wired up for this case). `"fp32"` for debugging.
