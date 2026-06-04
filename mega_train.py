"""
mega_train.py — Busca mais agressiva por modelo visando melhorar F1-macro
contra o consenso geral, sem perder as comparações contra médicas e pares.
"""

import json
import pickle
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier
from scipy.signal import welch
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import features as features_v2

ROOT = Path(__file__).parent
DATASET_PKL = ROOT / "dataset.pkl"
OUT_MODEL = ROOT / "model_mega.pkl"
OUT_REPORT = ROOT / "mega_report.txt"
OUT_RESULTS = ROOT / "mega_results.json"
CACHE_V1 = ROOT / "mega_features_v1.pkl"
CACHE_V2 = ROOT / "mega_features_v2.pkl"

RANDOM_SEED = 42
TEST_SIZE = 0.20
RATERS = ["elaine", "amanda", "marina"]
BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 70.0),
}
NPERSEG = 256

warnings.filterwarnings("ignore")


def patient_dominant_label(meta: list[dict]) -> dict[str, str]:
    per_patient = defaultdict(list)
    for row in meta:
        per_patient[row["patient_id"]].append(row["label"])
    return {patient: Counter(labels).most_common(1)[0][0] for patient, labels in per_patient.items()}


def stratified_patient_split(
    patients: list[str],
    dominant: dict[str, str],
    test_size: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for patient in patients:
        by_label[dominant[patient]].append(patient)

    train_pats, test_pats = [], []
    for patient_list in by_label.values():
        patient_list = sorted(patient_list)
        rng.shuffle(patient_list)
        n_test = max(1, round(len(patient_list) * test_size))
        test_pats.extend(patient_list[:n_test])
        train_pats.extend(patient_list[n_test:])
    return sorted(train_pats), sorted(test_pats)


def _band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return 0.0
    return float(np.mean(psd[mask]))


def extract_window_v1(signal: np.ndarray, sfreq: float) -> np.ndarray:
    feats = []
    for ch in range(signal.shape[0]):
        x = signal[ch].astype(np.float64)
        freqs, psd = welch(x, fs=sfreq, nperseg=NPERSEG, window="hann")
        total = float(np.sum(psd)) + 1e-12
        band_feats = [
            _band_power(freqs, psd, lo, hi) / total
            for (lo, hi) in [BANDS[b] for b in BAND_ORDER]
        ]
        variance = float(np.var(x))
        zcr = float(np.sum(np.diff(np.sign(x)) != 0)) / len(x)
        feats.extend(band_feats + [variance, zcr])
    return np.array(feats, dtype=np.float32)


def build_feature_matrix(data: list[dict], mode: str) -> np.ndarray:
    cache_path = CACHE_V1 if mode == "v1" else CACHE_V2
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    rows = []
    for idx, row in enumerate(data):
        if idx % 1000 == 0:
            print(f"  features {mode}: {idx}/{len(data)}", flush=True)
        if mode == "v1":
            rows.append(extract_window_v1(row["signal"], row["sfreq"]))
        else:
            rows.append(features_v2.extract_window(row["signal"], row["sfreq"]))
    X = np.array(rows, dtype=np.float32)
    with open(cache_path, "wb") as f:
        pickle.dump(X, f, protocol=4)
    return X


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def agreement_weight(row: dict) -> float:
    labels = [row["labels_per_rater"].get(r, "normal") for r in RATERS]
    unique = len(set(labels))
    if unique == 1:
        return 1.4
    if unique == 2:
        return 1.0
    return 0.65


def class_weights(y_train: np.ndarray, scheme: str) -> dict[str, float]:
    counts = Counter(y_train)
    n_classes = len(counts)
    total = len(y_train)
    inv = {label: total / (n_classes * count) for label, count in counts.items()}
    if scheme == "balanced":
        return inv
    if scheme == "sqrt_balanced":
        return {label: float(np.sqrt(weight)) for label, weight in inv.items()}
    if scheme == "manual_focus":
        return {
            "gpd": 1.2,
            "grda": 10.0,
            "lpd": 3.0,
            "lrda": 15.0,
            "normal": 1.0,
            "other": 1.5,
            "seizure": 2.2,
        }
    return {label: 1.0 for label in counts}


def sample_weights(y_train: np.ndarray, meta_train: list[dict], scheme: str, agreement_scheme: str) -> np.ndarray:
    cw = class_weights(y_train, scheme)
    weights = []
    for label, row in zip(y_train, meta_train):
        weight = cw.get(label, 1.0)
        if agreement_scheme == "use":
            weight *= agreement_weight(row)
        weights.append(weight)
    return np.array(weights, dtype=np.float32)


def build_model(config: dict, num_classes: int):
    if config["model_type"] == "lgbm":
        return LGBMClassifier(
            objective="multiclass",
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            num_leaves=config["num_leaves"],
            max_depth=config["max_depth"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            min_child_samples=config["min_child_samples"],
            reg_alpha=config["reg_alpha"],
            reg_lambda=config["reg_lambda"],
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
        )
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        min_child_weight=config["min_child_weight"],
        reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"],
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def fit_predict_proba(config, X_train, y_train_enc, meta_train, X_valid, num_classes):
    weights = sample_weights(
        np.array([config["label_decoder"][idx] for idx in y_train_enc]),
        meta_train,
        config["class_weight_scheme"],
        config["agreement_scheme"],
    )

    if config.get("oversample"):
        ros = RandomOverSampler(random_state=RANDOM_SEED)
        X_train, y_train_enc = ros.fit_resample(X_train, y_train_enc)
        weights = np.ones(len(y_train_enc), dtype=np.float32)

    model = build_model(config, num_classes)
    model.fit(X_train, y_train_enc, sample_weight=weights)
    proba = model.predict_proba(X_valid)
    return np.asarray(proba), model


def candidate_configs(label_decoder: dict[int, str]):
    base = {"label_decoder": label_decoder}
    return [
        {
            **base, "name": "lgbm_v1_agree_manual", "feature_set": "v1", "model_type": "lgbm",
            "class_weight_scheme": "manual_focus", "agreement_scheme": "use",
            "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1,
            "subsample": 0.9, "colsample_bytree": 0.8, "min_child_samples": 20,
            "reg_alpha": 0.0, "reg_lambda": 0.0,
        },
        {
            **base, "name": "lgbm_v1_sqrt", "feature_set": "v1", "model_type": "lgbm",
            "class_weight_scheme": "sqrt_balanced", "agreement_scheme": "use",
            "n_estimators": 500, "learning_rate": 0.04, "num_leaves": 47, "max_depth": -1,
            "subsample": 0.85, "colsample_bytree": 0.85, "min_child_samples": 18,
            "reg_alpha": 0.05, "reg_lambda": 0.2,
        },
        {
            **base, "name": "lgbm_v1_manual_noagree", "feature_set": "v1", "model_type": "lgbm",
            "class_weight_scheme": "manual_focus", "agreement_scheme": "none",
            "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1,
            "subsample": 0.9, "colsample_bytree": 0.8, "min_child_samples": 20,
            "reg_alpha": 0.0, "reg_lambda": 0.0,
        },
    ]


def evaluate_consensus(y_true, y_pred, classes):
    return {
        cls: {
            "precision": precision_score((y_true == cls).astype(int), (y_pred == cls).astype(int), zero_division=0),
            "recall": recall_score((y_true == cls).astype(int), (y_pred == cls).astype(int), zero_division=0),
            "f1": f1_score((y_true == cls).astype(int), (y_pred == cls).astype(int), zero_division=0),
            "support": int((y_true == cls).sum()),
        }
        for cls in classes
    }


def pair_consensus_metrics(y_pred, meta_test):
    results = {}
    for r1, r2 in [("elaine", "amanda"), ("elaine", "marina"), ("amanda", "marina")]:
        agreed_idx = [
            idx for idx, row in enumerate(meta_test)
            if row["labels_per_rater"].get(r1, "normal") == row["labels_per_rater"].get(r2, "normal")
        ]
        y_true = np.array([meta_test[idx]["labels_per_rater"].get(r1, "normal") for idx in agreed_idx])
        y_hat = y_pred[agreed_idx]
        classes = sorted(set(y_true)) if len(y_true) else []
        macro = f1_score(y_true, y_hat, average="macro", labels=classes, zero_division=0) if classes else 0.0
        results[f"{r1}_{r2}"] = {
            "coverage": len(agreed_idx),
            "macro_f1": macro,
            "classes": classes,
        }
    return results


def run():
    print(f"Carregando {DATASET_PKL} ...")
    data = pickle.load(open(DATASET_PKL, "rb"))
    meta = [
        {
            "exam_id": row["exam_id"],
            "patient_id": row["patient_id"],
            "window_idx": row["window_idx"],
            "window_start": row["window_start"],
            "window_end": row["window_end"],
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

    y_train = y[train_idx]
    y_test = y[test_idx]
    meta_train = [meta[i] for i in train_idx]
    meta_test = [meta[i] for i in test_idx]

    le = LabelEncoder()
    le.fit(y)
    label_decoder = {idx: label for idx, label in enumerate(le.classes_)}

    configs = candidate_configs(label_decoder)
    needed_feature_sets = sorted({config["feature_set"] for config in configs})
    print("Construindo matrizes de features ...")
    feature_sets = {name: build_feature_matrix(data, name) for name in needed_feature_sets}
    print(f"Testando {len(configs)} configurações ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    all_results = []
    for config in configs:
        print(f"\n### {config['name']} ###")
        X_full = feature_sets[config["feature_set"]]
        X_train_raw = X_full[train_idx]
        y_train_enc = le.transform(y_train)
        fold_scores = []
        for fold, (tr, va) in enumerate(skf.split(X_train_raw, y_train_enc), start=1):
            X_tr_raw, X_va_raw = X_train_raw[tr], X_train_raw[va]
            X_tr, X_va, _, _ = standardize_train_test(X_tr_raw, X_va_raw)
            proba, _ = fit_predict_proba(
                config,
                X_tr,
                y_train_enc[tr],
                [meta_train[i] for i in tr],
                X_va,
                len(le.classes_),
            )
            preds = le.inverse_transform(np.argmax(proba, axis=1))
            score = f1_score(y_train[va], preds, average="macro", zero_division=0)
            fold_scores.append(score)
            print(f"  fold {fold}: {score:.4f}")

        cv_mean = float(np.mean(fold_scores))
        cv_std = float(np.std(fold_scores))
        print(f"  CV macro-F1: {cv_mean:.4f} ± {cv_std:.4f}")
        all_results.append({**config, "cv_mean": cv_mean, "cv_std": cv_std})

    all_results.sort(key=lambda item: item["cv_mean"], reverse=True)
    best = all_results[0]
    print(f"\nMelhor configuração: {best['name']} ({best['cv_mean']:.4f})")

    X_best = feature_sets[best["feature_set"]]
    X_train_raw = X_best[train_idx]
    X_test_raw = X_best[test_idx]
    X_train, X_test, mean, std = standardize_train_test(X_train_raw, X_test_raw)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    proba, model = fit_predict_proba(
        best,
        X_train,
        y_train_enc,
        meta_train,
        X_test,
        len(le.classes_),
    )
    y_pred = le.inverse_transform(np.argmax(proba, axis=1))
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    consensus = evaluate_consensus(y_test, y_pred, list(le.classes_))
    pair_metrics = pair_consensus_metrics(y_pred, meta_test)
    individual_metrics = {}
    for rater in RATERS:
        y_r = np.array([row["labels_per_rater"].get(rater, "normal") for row in meta_test])
        individual_metrics[rater] = {
            "macro_f1": f1_score(y_r, y_pred, average="macro", zero_division=0),
        }

    kappas = {
        f"{r1}_{r2}": cohen_kappa_score(
            [row["labels_per_rater"].get(r1, "normal") for row in meta_test],
            [row["labels_per_rater"].get(r2, "normal") for row in meta_test],
        )
        for r1, r2 in [("elaine", "amanda"), ("elaine", "marina"), ("amanda", "marina")]
    }

    report = {
        "best_config": best["name"],
        "feature_set": best["feature_set"],
        "cv_mean": best["cv_mean"],
        "cv_std": best["cv_std"],
        "test_macro_f1_consensus": macro_f1,
        "consensus_per_class": consensus,
        "pair_metrics": pair_metrics,
        "individual_metrics": individual_metrics,
        "inter_rater_kappa": kappas,
        "test_patients": test_pats,
    }

    with open(OUT_MODEL, "wb") as f:
        pickle.dump({
            "model": model,
            "label_encoder": le,
            "feature_set": best["feature_set"],
            "feature_mean": mean,
            "feature_std": std,
            "config": best,
            "test_patients": test_pats,
            "train_patients": train_pats,
        }, f, protocol=4)

    OUT_RESULTS.write_text(json.dumps(report, indent=2))

    lines = []
    lines.append("MEGA TRAIN REPORT")
    lines.append("=================")
    lines.append(f"Best config: {best['name']}")
    lines.append(f"Feature set: {best['feature_set']}")
    lines.append(f"CV macro-F1: {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
    lines.append(f"Test macro-F1 vs consensus: {macro_f1:.4f}")
    lines.append("")
    lines.append("Per-class F1 vs consensus:")
    for cls in le.classes_:
        lines.append(f"  {cls:10s} {consensus[cls]['f1']:.3f} (support={consensus[cls]['support']})")
    lines.append("")
    lines.append("Pairwise agreement subsets:")
    for pair, info in pair_metrics.items():
        lines.append(f"  {pair:16s} macro-F1={info['macro_f1']:.4f} coverage={info['coverage']}")
    lines.append("")
    lines.append("AI vs individual raters:")
    for rater, info in individual_metrics.items():
        lines.append(f"  {rater:10s} macro-F1={info['macro_f1']:.4f}")
    lines.append("")
    lines.append("Inter-rater kappa:")
    for pair, kappa in kappas.items():
        lines.append(f"  {pair:16s} kappa={kappa:.4f}")
    OUT_REPORT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    run()
