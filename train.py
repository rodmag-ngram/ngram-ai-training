"""
train.py — Treina LightGBM com split 80/20 por paciente.

Mantém o split estratificado por label dominante do paciente, faz CV de 5 folds
no conjunto de treino e salva `model.pkl` + `test_exams.txt`.
"""

import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent
FEAT_PKL = ROOT / "features.pkl"
MODEL_PKL = ROOT / "model.pkl"
TEST_EXAMS = ROOT / "test_exams.txt"

RANDOM_SEED = 42
TEST_SIZE = 0.20
CLASS_WEIGHT = {
    "gpd": 1.0,
    "grda": 8.0,
    "lpd": 1.0,
    "lrda": 15.0,
    "normal": 1.0,
    "other": 1.0,
    "seizure": 2.0,
}

BANDS = ["delta", "theta", "alpha", "beta", "gamma", "variance", "zcr"]
CH_NAMES = [
    "FP1", "F7", "T3", "T5", "O1", "F3", "C3", "P3", "FZ", "CZ", "PZ",
    "FP2", "F8", "T4", "T6", "O2", "F4", "C4", "P4", "CA1", "CA2", "CA3",
]
HEMISPHERE_PAIRS = ["FP1_FP2", "F3_F4", "F7_F8", "C3_C4", "P3_P4", "O1_O2", "T3_T4", "T5_T6"]
REGIONS = ["frontal", "temporal", "central", "parietal", "occipital"]


def patient_dominant_label(meta: list[dict]) -> dict[str, str]:
    pat_labels = defaultdict(list)
    for row in meta:
        pat_labels[row["patient_id"]].append(row["label"])
    return {patient: Counter(labels).most_common(1)[0][0] for patient, labels in pat_labels.items()}


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
    for patients_in_label in by_label.values():
        shuffled = sorted(patients_in_label)
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_size))
        test_pats.extend(shuffled[:n_test])
        train_pats.extend(shuffled[n_test:])

    return sorted(train_pats), sorted(test_pats)


def build_model() -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def feature_name(index: int) -> str:
    per_channel_count = len(CH_NAMES) * len(BANDS)
    if index < per_channel_count:
        channel_idx = index // len(BANDS)
        feat_idx = index % len(BANDS)
        return f"{CH_NAMES[channel_idx]}_{BANDS[feat_idx]}"

    index -= per_channel_count
    pair_band_count = len(HEMISPHERE_PAIRS) * 5
    if index < pair_band_count:
        pair_idx = index // 5
        band_idx = index % 5
        return f"asym_{HEMISPHERE_PAIRS[pair_idx]}_{BANDS[band_idx]}"

    index -= pair_band_count
    region_band_count = len(REGIONS) * 5
    if index < region_band_count:
        region_idx = index // 5
        band_idx = index % 5
        return f"region_{REGIONS[region_idx]}_{BANDS[band_idx]}"

    extra_names = [
        "corr_left_mean",
        "corr_right_mean",
        "corr_interhemispheric_mean",
        "ratio_slow_fast",
        "ratio_delta_alpha",
        "ratio_frontalslow_posterioralpha",
    ]
    if index < region_band_count + len(extra_names):
        return extra_names[index - region_band_count]
    return f"feature_{index}"


def main():
    print(f"Carregando {FEAT_PKL} ...")
    with open(FEAT_PKL, "rb") as f:
        bundle = pickle.load(f)
    X, y, meta = bundle["X"], bundle["y"], bundle["meta"]

    dominant = patient_dominant_label(meta)
    patients = sorted(dominant.keys())
    print(f"Pacientes totais: {len(patients)}")

    train_pats, test_pats = stratified_patient_split(
        patients, dominant, TEST_SIZE, RANDOM_SEED
    )
    print(f"Treino: {len(train_pats)} pacientes  |  Teste: {len(test_pats)} pacientes")
    print(f"  Teste (labels dominantes): {{ {', '.join(f'{p}: {dominant[p]}' for p in test_pats)} }}")

    train_idx = [i for i, row in enumerate(meta) if row["patient_id"] in set(train_pats)]
    test_idx = [i for i, row in enumerate(meta) if row["patient_id"] in set(test_pats)]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"\nJanelas treino: {len(X_train)}  |  Janelas teste: {len(X_test)}")
    print(f"Distribuição treino: {dict(Counter(y_train))}")
    print(f"Distribuição teste:  {dict(Counter(y_test))}")

    le = LabelEncoder()
    le.fit(y)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    print(f"\nClasses: {list(le.classes_)}")

    sample_weight = np.array([CLASS_WEIGHT.get(label, 1.0) for label in y_train], dtype=np.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    print("\nCross-validation (5-fold) no conjunto de treino ...")
    cv_f1s = []
    for fold, (tr, val) in enumerate(skf.split(X_train, y_train_enc), start=1):
        model = build_model()
        model.fit(X_train[tr], y_train_enc[tr], sample_weight=sample_weight[tr])
        preds = model.predict(X_train[val])
        f1 = f1_score(y_train_enc[val], preds, average="macro")
        cv_f1s.append(f1)
        print(f"  Fold {fold}: F1-macro = {f1:.4f}")
    print(f"\nCV F1-macro: {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")

    print("\nTreinando modelo final nos 80% completos ...")
    model = build_model()
    model.fit(X_train, y_train_enc, sample_weight=sample_weight)

    preds_test = model.predict(X_test)
    f1_test = f1_score(y_test_enc, preds_test, average="macro")
    print(f"\nF1-macro no conjunto de teste: {f1_test:.4f}")
    print("\nRelatório por classe:")
    print(classification_report(
        y_test_enc,
        preds_test,
        target_names=le.classes_,
        digits=3,
        zero_division=0,
    ))

    importance = model.feature_importances_
    top10 = np.argsort(importance)[::-1][:10]
    print("\nTop-10 features por importância:")
    for rank, feature_idx in enumerate(top10, start=1):
        print(f"  {rank:2d}. {feature_name(feature_idx):20s}  {importance[feature_idx]:.4f}")

    test_exam_ids = sorted({meta[i]["exam_id"] for i in test_idx})
    print(f"\nSalvando {MODEL_PKL} ...")
    with open(MODEL_PKL, "wb") as f:
        pickle.dump({
            "model": model,
            "model_type": "LightGBM",
            "class_weight": CLASS_WEIGHT,
            "label_encoder": le,
            "mu": bundle["mu"],
            "std": bundle["std"],
            "test_patients": test_pats,
            "train_patients": train_pats,
        }, f, protocol=4)

    print(f"Salvando {TEST_EXAMS} ({len(test_exam_ids)} exames) ...")
    TEST_EXAMS.write_text("\n".join(test_exam_ids) + "\n")
    print("\nPronto.")


if __name__ == "__main__":
    main()
