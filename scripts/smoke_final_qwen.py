import os

import joblib

from scripts.finalize_customer_adaptation import load_all


os.environ.pop("QWEN_LOCAL_MODEL_DIR", None)
os.environ["QWEN_DEVICE"] = "cuda"

bundle = joblib.load("models/customer_2026-09-21/category_qwen_finetuned.joblib")
row = load_all()[0][0]
probabilities = bundle["pipeline"].predict_proba([row])
print("QWEN_END_TO_END_OK", probabilities.shape, round(float(probabilities.sum()), 6))
