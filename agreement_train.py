"""
agreement_train.py — Experimentos treinando em subsets filtrados por acordo médico.
"""

import json
import pickle
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from mega_train import (
    build_feature_matrix,
    evaluate_consensus,
    pair_consensus_metrics,
    patient_dominant_label,
    standardize_train_test,
    stratified_patient_split,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
DATASET_PKL = ROOT / "dataset.pkl"
OUT_JSON = ROOT / "agreement_results.json"
OUT_TXT = ROOT / "agreement_report.txt"
RATERS = ["elaine", "amanda", "marina"]
RANDOM_SEED = 42
TEST_SIZE = 0.20


def agreement_level(row: dict) -> str:
    labels = [row["labels_per_rater"].get(r, "normal") for r in RATERS]
    uniq = len(set(labels))
    if uniq == 1:
        return "3of3"
    if uniq == 2:
        return "2of3"
    return "1of3"


def filtered_sample_weight(row: dict, mode: str) -> float:
    level = agreement_level(row)
    if mode == "full":
        return 1.0
    if mode == "weighted":
        return {"3of3": 1.5, "2of3": 1.0, "1of3": 0.4}[level]
    if mode == "strict":
        return 1.0 if level == "3of3" else 0.0
    if mode == "semi_strict":
        return 1.0 if level in {"3of3", "2of3"} else 0.0
    if mode == "hybrid":
        return {"3of3": 1.7, "2of3": 1.0, "1of3": 0.2}[level]
    return 1.0


def class_weight(label: str) -> float:
    return {
        "gpd": 1.2,
        "grda": 8.0,
        "lpd": 3.0,
        "lrda": 10.0,
        "normal": 1.0,
        "other": 1.4,
        "seizure": 1.8,
    }.get(label, 1.0)


def build_model():
    return LGBMClassifier(
        objective="multiclass",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def train_mode(mode: str, X_train_raw, y_train, meta_train, X_test_raw, y_test, meta_test, le):
    y_train_enc = le.transform(y_train)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = []

    for tr, va in skf.split(X_train_raw, y_train_enc):
        X_tr, X_va, _, _ = standardize_train_test(X_train_raw[tr], X_train_raw[va])
        weights = np.array([
            class_weight(y_train[i]) * filtered_sample_weight(meta_train[i], mode)
            for i in tr
        ], dtype=np.float32)
        keep = weights > 0
        X_fit = X_tr[keep]
        y_fit = y_train_enc[tr][keep]
        w_fit = weights[keep]
        model = build_model()
        model.fit(X_fit, y_fit, sample_weight=w_fit)
        preds = le.inverse_transform(np.asarray(model.predict(X_va), dtype=int))
        cv_scores.append(f1_score(y_train[va], preds, average="macro", zero_division=0))

    X_train, X_test, mean, std = standardize_train_test(X_train_raw, X_test_raw)
    weights = np.array([
        class_weight(label) * filtered_sample_weight(row, mode)
        for label, row in zip(y_train, meta_train)
    ], dtype=np.float32)
    keep = weights > 0
    model = build_model()
    model.fit(X_train[keep], y_train_enc[keep], sample_weight=weights[keep])
    preds = le.inverse_transform(np.asarray(model.predict(X_test), dtype=int))

    consensus = evaluate_consensus(y_test, preds, list(le.classes_))
    pair_metrics = pair_consensus_metrics(preds, meta_test)
    indiv = {}
    for rater in RATERS:
        y_r = np.array([row["labels_per_rater"].get(rater, "normal") for row in meta_test])
        indiv[rater] = f1_score(y_r, preds, average="macro", zero_division=0)

    return {
        "mode": mode,
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "test_macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "consensus": consensus,
        "pairs": pair_metrics,
        "individual": indiv,
        "train_kept": int(keep.sum()),
        "model": model,
        "mean": mean,
        "std": std,
    }


def main():
    data = pickle.load(open(DATASET_PKL, "rb"))
    meta = [
        {
            "exam_id": row["exam_id"],
            "patient_id": row["patient_id"],
            "window_idx": row["window_idx"],
            "label": row["label"],
            "labels_per_rater": row["labels_per_rater"],
        }
        for row in data
    ]
    y = np.array([row["label"] for row in meta])
    patients = sorted(patient_dominant_label(meta).keys())
    dominant = patient_dominant_label(meta)
    train_pats, test_pats = stratified_patient_split(patients, dominant, TEST_SIZE, RANDOM_SEED)
    train_idx = [i for i, row in enumerate(meta) if row["patient_id"] in set(train_pats)]
    test_idx = [i for i, row in enumerate(meta) if row["patient_id"] in set(test_pats)]

    X = build_feature_matrix(data, "v1")
    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    meta_train = [meta[i] for i in train_idx]
    meta_test = [meta[i] for i in test_idx]

    le = LabelEncoder()
    le.fit(y)

    modes = ["full", "weighted", "semi_strict", "strict", "hybrid"]
    results = []
    for mode in modes:
        print(f"\n### {mode} ###", flush=True)
        result = train_mode(mode, X_train_raw, y_train, meta_train, X_test_raw, y_test, meta_test, le)
        print(
            f"cv={result['cv_mean']:.4f} test={result['test_macro_f1']:.4f} kept={result['train_kept']}",
            flush=True,
        )
        results.append(result)

    best = max(results, key=lambda item: item["test_macro_f1"])
    kappas = {
        f"{r1}_{r2}": cohen_kappa_score(
            [row["labels_per_rater"].get(r1, "normal") for row in meta_test],
            [row["labels_per_rater"].get(r2, "normal") for row in meta_test],
        )
        for r1, r2 in [("elaine", "amanda"), ("elaine", "marina"), ("amanda", "marina")]
    }

    payload = {
        "best_mode": best["mode"],
        "results": [
            {
                "mode": res["mode"],
                "cv_mean": res["cv_mean"],
                "cv_std": res["cv_std"],
                "test_macro_f1": res["test_macro_f1"],
                "train_kept": res["train_kept"],
                "individual": res["individual"],
                "pairs": res["pairs"],
            }
            for res in results
        ],
        "best_consensus": best["consensus"],
        "best_pairs": best["pairs"],
        "best_individual": best["individual"],
        "inter_rater_kappa": kappas,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    lines = []
    lines.append("AGREEMENT TRAIN REPORT")
    lines.append("======================")
    for res in results:
        lines.append(
            f"{res['mode']:12s} cv={res['cv_mean']:.4f}±{res['cv_std']:.4f} "
            f"test={res['test_macro_f1']:.4f} kept={res['train_kept']}"
        )
    lines.append("")
    lines.append(f"Best mode: {best['mode']}")
    lines.append("Per-class F1 vs consensus:")
    for cls in le.classes_:
        lines.append(f"  {cls:10s} {best['consensus'][cls]['f1']:.3f}")
    lines.append("")
    lines.append("Pairwise agreement subsets:")
    for pair, info in best["pairs"].items():
        lines.append(f"  {pair:16s} macro-F1={info['macro_f1']:.4f} coverage={info['coverage']}")
    lines.append("")
    lines.append("AI vs individual raters:")
    for rater, score in best["individual"].items():
        lines.append(f"  {rater:10s} macro-F1={score:.4f}")
    OUT_TXT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
