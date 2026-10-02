from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/gpu_research_v4/synthetic"
REJECTED_CATEGORIES = {
    "Прочее": "Residual class has no coherent positive concept; synthetic positives would teach arbitrary wording.",
    "Электронные обращения": "Channel/template cues are shared by many semantic classes; generated positives are not diagnostic enough.",
    "Проблема из почты": "The tiny real anchor is channel-dependent and semantically heterogeneous; generated mail-channel text is unsafe.",
}
SPECIAL_NOTES = {
    "Тарификация посылок": (
        "Approved only as SYNTHETIC_ONLY at weight <=0.05. Definition is taxonomy/official-document based; "
        "no validation/test example was inspected. Metrics must remain separate."
    ),
    "Критический инцидент (КИ)": "Single real anchor retained; generated cases add impact/scale diversity and use weight <=0.10.",
    "Массовая отправка": "Single real anchor retained; samples vary batch-operation situations rather than wording only; weight <=0.10.",
    "Не выгружается бланк": "Single real anchor retained; payment-vs-document boundary is explicit; weight <=0.10.",
    "Событие мониторинга ИС": "Single real anchor retained; internal-IS-vs-general-monitoring boundary is explicit; weight <=0.10.",
}
REJECTED_SAMPLES = {
    "b97dd930e673c2d20c3bffe377eeae75d94288e8e6e2b429fbaf4e17c60abc51": "Payment-system failure competes with the form-download signal.",
    "20715a097c8ba37b20121e2c4c1de40d7e1e0165cfd6ca6fbe4594569880216b": "Missing bonuses is a stronger neighboring-class signal than the generic Android error.",
    "09855f7a3b74abf37f573e5c2423db300388bf014d8c1271f43b2b39b4fea637": "Empty form-103 archive adds an unrelated import failure to the tracking event.",
    "0742779e0b0540942db701987de960d0c972e412c33532833d7503c60cc35379": "Explicitly contains several unrelated questions and is not a clean change request.",
    "fac2ba40ec64fcc56c88bd40234912f29dabeee9631f4ba6fbeef2feec69621d": "Application crash and missing QR are two competing incident objects.",
    "af57e9c655f788be37ebf84e3bd0b5bf01d6707400a6160172a10ea892d79758": "Missing refund competes with the electronic-authority signal.",
    "07057dbba41e2a4beed2ff4cdd640ea33df1c9291f919965cfc224845543bc2d": "Missing contract competes with the address-directory failure.",
    "2f8bad730e4a304c74d548b5ebd9fa98a54647f6a88b8780d4dbb98136d15c38": "Authentication failure competes with missing organization data.",
    "e3bafbe597e9963fd38ed4046c99a353859f36d1f7d87889f7a1b7bd063dbbd2": "Mobile application crash competes with document formation.",
}


def main() -> None:
    pack = json.loads((ART / "domain_corpus_audit_pack.json").read_text(encoding="utf-8"))
    reviews = []
    approved = []
    rejected_hashes = sorted(REJECTED_SAMPLES)
    for item in pack["classes"]:
        label = str(item["category"])
        representatives = item["representative_samples"]
        reviewed_hashes = [str(sample["sample_sha256"]) for sample in representatives]
        if label in REJECTED_CATEGORIES:
            decision = "REJECT_CLASS"
            note = REJECTED_CATEGORIES[label]
        else:
            decision = "APPROVE_FILTERED_CLASS"
            approved.append(label)
            note = SPECIAL_NOTES.get(
                label,
                (
                    "Low/median/high-margin representatives were reviewed against the real-TRAIN style profile, "
                    "category meaning and listed confusion neighbors. The intended diagnostic object remains explicit; "
                    "only the automatically filtered diverse corpus is eligible."
                ),
            )
        reviews.append(
            {
                "category": label,
                "decision": decision,
                "reviewed_representative_sha256": reviewed_hashes,
                "representative_count": len(representatives),
                "real_train_support": item["real_train_support"],
                "confusable_categories_checked": item["confusable_categories"],
                "notes": note,
            }
        )
    result = {
        "status": "COMPLETE",
        "reviewer": "Codex synthetic-data architect/auditor",
        "review_scope": (
            "Every generated class: low-, median- and high-confusion-margin representatives after automatic QC; "
            "also a prior three-per-class raw-corpus review."
        ),
        "approved_categories": approved,
        "rejected_categories": sorted(REJECTED_CATEGORIES),
        "rejected_sample_sha256": rejected_hashes,
        "rejected_sample_reasons": REJECTED_SAMPLES,
        "summary": {
            "classes_reviewed": len(reviews),
            "classes_approved": len(approved),
            "classes_rejected": len(REJECTED_CATEGORIES),
            "representatives_reviewed": sum(item["representative_count"] for item in reviews),
            "representatives_rejected": len(rejected_hashes),
            "policy": "Automatic QC is necessary but insufficient; class-level semantic approval gates training.",
        },
        "class_reviews": reviews,
    }
    (ART / "manual_synthetic_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path = ART / "domain_corpus_clean_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "CLEAN_SEMANTIC_AUDIT_COMPLETE"
    manifest["manual_semantic_audit"] = str(
        (ART / "manual_synthetic_audit.json").relative_to(ROOT)
    )
    manifest["manual_semantic_audit_summary"] = result["summary"]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
