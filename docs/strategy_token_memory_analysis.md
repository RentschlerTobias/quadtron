# Tokenization Strategy Comparison: Memory & Token Analysis

## Strategies Overview

| Strategy | Description | Token Compression |
|----------|-------------|-------------------|
| **S0** | Lexicographical sorting (baseline) | No compression |
| **S1** | Directed row traversal (full 8 tokens/face) | No compression |
| **S2** | Row-compressed emission (EOR token) | Face pairs share edge tokens |
| **S3** | Like S2 but rows enforced left-to-right (all CW) | Same as S2 |

## Token Structure

### Per-Face Token Count

| Strategy | Tokens/Face | Explanation |
|----------|-------------|-------------|
| **S0, S1** | 8 coords (4 vertices × 2 coords) | Full representation |
| **S2, S3** | 4 coords (first face) + 2 coords (subsequent faces) | First face in row fully, then only shared edge |

### Token Sequence Structure

```
S0/S1: [START×8] [v0_y,v0_x, v1_y,v1_x, v2_y,v2_x, v3_y,v3_x] × n_faces [END×8]
         16 tokens              8 tokens per face × n_faces            16 tokens

S2/S3: [START×8] [row0_face0_full] [row0_face1_partial] ... [EOR]
                               [row1_face0_full] ... [EOR] ... [END×8]
```

### Special Tokens

| Token | Value | Count |
|-------|-------|-------|
| start_token | quantization_levels (=256) | 8 × n_rows |
| end_token | quantization_levels + 1 (=257) | 8 |
| eor_token | quantization_levels + 2 (=258) | 1 × n_rows |

## Token Count Analysis

### Per-Mesh Calculation (Example: 24 faces)

| Strategy | Face Tokens | EOR Tokens | Start/End | **Total** |
|----------|-------------|------------|-----------|-----------|
| **S0, S1** | 24 × 8 = 192 | 0 | 16 | **208** |
| **S2, S3** | ~116 + EOR | ~16 | 16 | **~132** |

### Dataset Aggregation (1229 Meshes)

| Metric | S0/S1 | S2/S3 | Reduction |
|--------|-------|-------|-----------|
| **avg tokens/mesh** | 328.1 | 197.4 | **-40%** |
| **min tokens** | 208 | 132 | -37% |
| **max tokens** | 448 | 272 | -39% |
| **tokens/face** | 8.41 | 5.06 | -40% |
| **avg faces/mesh** | 39.0 | 39.0 | 0% |

## Memory Analysis

### Attention Complexity

Attention mechanism has quadratic complexity: **O(seq_len²)**

For cross-attention between input tokens and latent queries, the sequence length directly impacts memory consumption.

| max_seq_length | Attention Activation (bf16) | Relative |
|----------------|----------------------------|----------|
| 448 (S0/S1) | ~0.38 MB | 100% |
| 272 (S2/S3) | ~0.14 MB | **-63%** |

### Observed GPU Memory from Training Runs

| Strategy | batch_size | Peak Memory | Notes |
|----------|------------|-------------|-------|
| S1 | 48 | ~15.9 GB | Full seq length |
| S1 (mem_matched) | 24 | ~8.3 GB | Halved batch |
| S2 | 48 | ~8.4 GB | Short seq compensates larger batch |

### Memory Match Explanation

Strategy 2 with batch_size=48 achieves similar GPU memory as Strategy 1 with batch_size=24:

- **S2**: max_seq_length=272 → less activation memory per sample
- **S1**: max_seq_length=448 → more activation memory per sample

This allows S2 to use **2× larger batch size** for the same memory footprint.

## Key Insights

1. **Token Efficiency**: S2/S3 reduce tokens per mesh by ~40% through row-compressed emission

2. **Memory Efficiency**: Shorter sequences (272 vs 448) allow:
   - ~60% less activation memory
   - 2× larger batch size for same memory

3. **Quality Trade-off**: Earlier comparison showed S2 had worse bpt (~1.8) vs S1 (~1.0), but this may be due to vertex ordering bugs that have now been fixed in S3

4. **S3 vs S2**: S3 maintains the same compression efficiency as S2 but with consistent vertex ordering (always CW, left-to-right), which may improve model learning

## Summary Table

| Aspect | S0 | S1 | S2 | S3 |
|--------|----|----|----|----|
| Token/face | 8 | 8 | ~5 | ~5 |
| Row direction | N/A | mixed | mixed | left-to-right |
| Vertex order | CCW | directed | directed | CW |
| Compression | None | None | EOR-based | EOR-based |
| Max seq len | 448 | 448 | 272 | 272 |
| Memory/batch | high | high | low | low |

## Recommendation

S3 (row-compressed + left-to-right CW) combines:
- **40% fewer tokens** → faster training, smaller models
- **60% less activation memory** → 2× batch size possible
- **Consistent vertex ordering** → potentially better model learning

This makes S3 the most memory-efficient strategy while maintaining structural consistency for the model to learn.