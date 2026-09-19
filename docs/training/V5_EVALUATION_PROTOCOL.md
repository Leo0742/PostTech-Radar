# PostTech Radar V5.1 — Frozen Evaluation Protocol

- Source XLSX SHA-256: `daf742da6bb5902149585a05a3e38a2c655347cf6001f9cae640e1f27069ac40`
- Canonical protocol dataset SHA-256: `419cf17fa83ad0c6c77c5b72ba0eda24455f0e3f14c2a8b4634669c518d0342e`
- Split SHA-256: `533cac2bfe745ee22df3051b0a3dc0765cb04eb02341a75f0418e16fc0b2b1ae`
- Real rows: **1931**
- Duplicate groups: **1830** unique groups; **89** groups contain >1 ticket.
- Conflicting-label duplicate groups: **32**.
- Repeated grouped CV: **3 repeats × 4 folds = 12 frozen folds**.
- Development rows: **1241**.
- Calibration rows: **306**.
- V5 INTERNAL LOCKBOX rows: **384**.
- Temporal train/test rows: **1552 / 379**.
- Group overlap audit: **PASS**.
- Synthetic validation rows: **0**; synthetic data is train-only.

## Rules

- Duplicate identity is lowercase + Unicode NFKC + whitespace normalization of `Описание 2`; empty descriptions fall back to request identity.
- Splits are group-disjoint. Repeated StratifiedGroupKFold uses fixed seeds 20260917, 20260918 and 20260919.
- Categories with fewer than five independent groups remain development-only; V5 reports them honestly instead of pretending a reliable internal lockbox estimate.
- The V4 sealed holdout is historical reference only. This new subset is explicitly named **V5 INTERNAL LOCKBOX**, not an unseen external test.
- The true future external test is future operator-confirmed production data.
