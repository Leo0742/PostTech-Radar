# Public model artifacts

The binary deployment bundles are **not committed to this public repository**.

The original project produced two local category artifacts:

| Profile | Local artifact | SHA-256 |
|---|---|---|
| 4B Lite | `models/v5/category_qwen4b_lite.joblib` | `ebd02b8d711b91a7b2ce63c30642d63a5e0a5a8320c2064f4be130c901550d26` |
| 8B Quality | `models/v5/category.joblib` | `5c02b1337590c4e59bc2c0bfd5f279a3baee6a0bfe6945de568265afdf010e98` |

I keep the binaries private because they were trained from provided Service Desk data and serialized pipelines can contain vocabulary or other derived information from that dataset.

What is public instead:

- the 4B/8B training and deployment builders in `scripts/`;
- public experiment configs in `configs/v5/`;
- model architecture/runtime code in `backend/app/ml/`;
- grouped split and leakage checks;
- aggregate training reports in `docs/training/`;
- artifact hashes above.

See [TRAINING.md](../../TRAINING.md) for the full training path.
