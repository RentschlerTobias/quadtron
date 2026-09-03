# sweep.db Study Summary
| Rank | Study | Trials | Best Val BPT | Parameters |
|---|---|---|---|---|
| 1 | All Parameters Combined | 26 | 0.378289 | d_model, n_stages, layer_0, layer_1, layer_2, layer_3, learning_rate, warmup_steps, dropout, weight_decay, n_latents, sorting_strategy |
| 2 | Stage 2 — Architecture Search | 12 | 0.394341 | d_model, stage_layers |
| 3 | Sorting Strategy Comparison | 67 | 0.401317 | sorting_strategy |
| 4 | Stage 3 — Hyperparameter Tuning | 2 | 0.420500 | learning_rate, warmup_steps, weight_decay, dropout, batch_size, accumulation_steps |
