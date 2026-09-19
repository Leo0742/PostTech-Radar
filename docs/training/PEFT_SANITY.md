# Qwen8B PEFT / LoRA sanity check

I tested whether adapting the Qwen3-Embedding-8B encoder itself could improve the frozen-encoder pipeline.

This was a real PEFT experiment, but **it was not selected for the final model**.

## Sanity result

The rank-16 LoRA sanity run had:

- about **15.34M trainable parameters**;
- about **0.2022%** of total parameters trainable;
- 20 optimization steps;
- loss: **1.60318 → 1.31175**;
- minimum loss: **1.16900**;
- mean embedding cosine drift: **0.01482**.

This showed that gradients flowed through the adapters and that the embeddings changed.

## Why I did not deploy it

After the sanity check I ran more serious grouped-fold comparisons. The PEFT variants did not show a stable improvement over the frozen Qwen8B encoder.

One softer SupCon check gave:

| Fold | PEFT Top-1 / Top-3 / Macro-F1 | Frozen Top-1 / Top-3 / Macro-F1 |
|---|---|---|
| 1 | 80.083 / 98.340 / 74.992 | 80.913 / 97.510 / 76.446 |
| 2 | 78.423 / 99.170 / 72.572 | 79.668 / 98.340 / 74.798 |

Because the gain was not stable, I stopped the broad PEFT search and kept the simpler frozen-encoder design.

The public runnable sanity code is in [`scripts/v5_2_peft_sanity.py`](../../scripts/v5_2_peft_sanity.py).
