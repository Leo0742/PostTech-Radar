# Qwen4B matched V5 recipe vs Qwen8B matched V5 recipe

This is a matched-model-size reference, not a re-tuning of Qwen4B.

- Instruction: en_concise
- Representation: separate_metadata_text
- Sequence length: 512
- Common embedding dimension: 2560
- Frozen 8B winning dimension before matching: 3072
- Head/calibration: calibrated_linearsvc / calibrated_linearsvc
- Synthetic recipe: None

| Metric | Qwen4B matched | Qwen8B matched | 8B - 4B |
|---|---:|---:|---:|
| TOP1 | 0.783648 | 0.781238 | -0.002410 |
| TOP3 | 0.971717 | 0.969298 | -0.002419 |
| Macro-F1 | 0.712706 | 0.710134 | -0.002572 |
| FULL43 Macro-F1 | 0.495123 | 0.509097 | +0.013974 |
| Benchmark end-to-end, s | 43.655 | 65.903 | — |
| Embedding latency, ms/row | 28.913 | 49.106 | — |
| Peak VRAM, MiB | 8639.5 | 15468.1 | — |

Latency/VRAM benchmark: same TOP15 repeat 0 / fold 0, fresh dedicated cache, batch size 8.
