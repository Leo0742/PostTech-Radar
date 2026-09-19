# PostTech Radar — Qwen3-Embedding-4B Lite Final

Date: 2026-09-18

## Final Lite profile

- Model: `Qwen/Qwen3-Embedding-4B`
- Exact revision: `5cf2132abc99cad020ac570b19d031efec650f2b`
- Max length: 512
- Embedding width: 2560
- Instruction: `posttech_tight_c`
- Representation: `separate_metadata_text`
- Metadata scale: 0.75
- Ensemble: 0.90 supervised + 0.05 prototype + 0.05 kNN
- kNN neighbors: 11
- Prototype temperature: 0.08
- Specialist: leakage-safe TF-IDF confusion-pair correction
- Full43 stable specialist pairs: 13
- Specialist threshold: 0.0
- Training rows: 1,931 real labeled tickets
- Synthetic deployment rows: 0
- Lockbox access during this sprint: none
- Artifact: `models/v5/category_qwen4b_lite.joblib`
- Artifact SHA256: `ebd02b8d711b91a7b2ce63c30642d63a5e0a5a8320c2064f4be130c901550d26`
- Artifact size: 34,615,629 bytes

The Full43 deployment dataset contains three labels with only one real example each. Standard cross-validated probability calibration cannot be fit for those classes. The deployment exporter therefore uses a balanced LogisticRegression fallback for the Full43 supervised head. TOP15 evaluation has enough class support for calibrated LinearSVC.

## Leakage-safe development quality

The specialist estimate below uses leave-one-repeat-out selection. Each held-out repeat is never used to choose its specialist pairs or threshold.

| View | TOP1 | TOP3 | Macro-F1 |
|---|---:|---:|---:|
| TOP15 base | 79.539% | 97.653% | 73.357% |
| TOP15 + leakage-safe specialist | **80.781%** | **97.653%** | **76.008%** |
| FULL43 base | 69.220% | 89.875% | 53.600% |
| FULL43 + leakage-safe specialist | **73.276%** | **89.875%** | **55.292%** |

Specialist error accounting:

- TOP15: 80 corrected errors, 44 introduced errors;
- FULL43: 278 corrected errors, 127 introduced errors.

The higher fit-on-all-development specialist numbers are not used as the headline quality estimate.

## 4B Lite vs 8B Quality

This table compares the two complete tuned recipes. It does not prove that the 4B encoder itself is better than 8B because the Lite pipeline received its own tuning.

| Metric | 4B Lite | 8B Quality | 4B delta |
|---|---:|---:|---:|
| TOP15 TOP1 | **80.781%** | 79.883% | **+0.898 pp** |
| TOP15 TOP3 | **97.653%** | 97.136% | **+0.517 pp** |
| TOP15 Macro-F1 | **76.008%** | 74.007% | **+2.001 pp** |
| FULL43 TOP1 | **73.276%** | 68.171% | **+5.105 pp** |
| FULL43 TOP3 | **89.875%** | 88.799% | **+1.076 pp** |
| FULL43 Macro-F1 | **55.292%** | 53.771% | **+1.521 pp** |
| Cold inference, RTX 3090 | **9.046 s** | 30.405 s | **70.25% lower** |
| Warm median, RTX 3090 | **0.090 s** | 0.120 s | **24.92% lower** |
| Peak VRAM | **7,838 MiB** | 14,562 MiB | **46.17% lower** |
| Artifact size | **34.62 MB** | 38.58 MB | **10.27% lower** |

The joblib artifact size does not include the separately cached Qwen encoder weights.
