# V5.2 Qwen8B Quality Max

Post-lockbox development branch: `V5.2_QWEN8B_QUALITY_MAX`.

Selection rule: DEVELOPMENT repeated grouped CV only. The V5 lockbox is historical evidence and is not used for V5.2 tuning or selection.

## Frozen V5 reference

| system | TOP15 TOP1 | TOP15 TOP3 | TOP15 Macro-F1 | FULL43 TOP1 | FULL43 TOP3 | FULL43 Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| V5_LOCKBOX_CHAMPION lockbox | 81.19% | 98.02% | 76.40% | 70.05% | 90.63% | 53.13% |
| V5 development baseline | 79.16% | 97.14% | 72.63% | 67.34% | 88.80% | 52.29% |

Frozen champion recipe: Qwen3-Embedding-8B, `posttech_tight_c`, `separate_metadata_text`, max_length 512, dim 3072, 0.90 calibrated LinearSVC + 0.05 prototype + 0.05 kNN.

## V5.2 experiment ledger

| candidate | TOP15 TOP1 | TOP15 TOP3 | TOP15 Macro-F1 | FULL43 Macro-F1 | status |
|---|---:|---:|---:|---:|---|
| current frozen V5 development champion | 79.16% | 97.14% | 72.63% | 52.29% | baseline |
| PEFT real-only | fold checks below | fold checks below | fold checks below | - | rejected: no stable gain |
| best TOP15 synthetic screen | 79.124% | 97.135% | 72.576% | - | rejected: below frozen; formal best used 0 synthetic rows |
| best OOF stack smoke | 71.49% | 95.04% | 68.02% | - | rejected |
| safe TF-IDF specialist | 79.883% | 97.136% | 74.007% | 53.771% | selected |
| final V5.2 deployment candidate | 79.883% | 97.136% | 74.007% | 53.771% | selected and deployed |

## Final development findings

- Matched Qwen3-Embedding-4B reference completed: 24/24 folds, no errors. TOP15 development mean TOP1 78.92%, TOP3 97.38%, Macro-F1 72.16%; FULL43 development mean TOP1 66.99%, TOP3 89.04%, Macro-F1 51.84%.
- Primary GPU work returned to Qwen3-Embedding-8B.
- Full 4096 vs 3072 quality check completed on 48/48 repeated grouped-CV runs, no errors and no lockbox access. Re-aggregated means from the preserved result files: 4096 TOP15 79.124/97.067/72.475 and FULL43 67.552/88.827/52.418; 3072 TOP15 79.158/97.135/72.625 and FULL43 67.338/88.801/52.293 (TOP1/TOP3/Macro-F1, %). Because TOP15 TOP1 and Macro-F1 are both slightly better at 3072, 3072 remains the primary deployment width.
- PEFT sanity passed for Qwen3-Embedding-8B LoRA rank16 (15.34M trainable parameters, 0.2022% of total): 20-step loss 1.60318 -> 1.31175, minimum 1.16900, mean embedding cosine drift 0.01482.
- Aggressive PEFT variants did not beat frozen. A softer SupCon rank16 check also failed to show stable improvement: fold1 80.083/98.340/74.992 vs frozen 80.913/97.510/76.446; fold2 78.423/99.170/72.572 vs frozen 79.668/98.340/74.798 (TOP1/TOP3/Macro-F1). Broad PEFT search was stopped.
- OOF stack smoke was materially worse at about 71.49% TOP1 / 95.04% TOP3 / 68.02% Macro-F1 and was rejected.
- The TF-IDF specialist originally exposed a forbidden target-like field (`routing_target`). That field was removed before final measurement. All specialist metrics below are leakage-safe and use only description, service, component, request_type, user, criticality, urgency and priority.
- Leakage-safe TOP15 specialist: 79.883% TOP1 / 97.136% TOP3 / 74.007% Macro-F1, about +0.72 pp TOP1 and +1.38 pp Macro-F1 vs frozen.
- Leakage-safe FULL43 specialist: 68.171% TOP1 / 88.799% TOP3 / 53.771% Macro-F1, about +0.83 pp TOP1 and +1.48 pp Macro-F1 vs frozen.
- Synthetic sweep completed over all 27 configurations per view on the frozen development folds. Synthetic rows were TRAIN-only, validation remained real-only, and the V5 lockbox was not accessed.
- TOP15 synthetic screen: formal best 79.124% TOP1 / 97.135% TOP3 / 72.576% Macro-F1, slightly below frozen. That formal best used 0 synthetic rows, so there is no validated TOP15 synthetic gain.
- FULL43 synthetic screen found 71.743% TOP1 / 89.070% TOP3 / 50.496% Macro-F1 for rare20_hard, ratio 2.0, weight 0.35. This improves FULL43 TOP1 but hurts Macro-F1, and synthetic is not selected because the primary TOP15 criterion did not improve.

## Final deployment

- Final recipe: Qwen3-Embedding-8B, revision `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`, `posttech_tight_c`, `separate_metadata_text`, max_length 512, dim 3072, 0.90 calibrated LinearSVC + 0.05 prototype + 0.05 kNN, plus the leakage-safe TF-IDF specialist.
- Trained on all eligible 1,931 real labeled tickets; synthetic_training_rows = 0.
- Deployment bundle: `models/v5/category.joblib`; category runtime reports `category_source=v5`.
- Incoming analysis payload now forwards `registration_date` and `user` (`user_name` -> `user`) into V5 structured features.
- Real API smoke passed through `/api/tickets/analyze`: full input, description-only partial input, and missing-text/metadata-only input all returned HTTP 200; category TOP-3 and routing outputs were present.
- Frontend root returned HTTP 200 and served HTML.
- Cold first inference after model load: about 13.9 s. Warm API inference: about 0.22-0.23 s per request on the benchmark GPU server.
- Local preserved artifacts: `outputs/final_v5_2/synthetic_screen.json` and `models/v5/category.joblib`.
