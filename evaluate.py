"""
evaluate.py — Compara AI com cada médica e médicas entre si nos exames de teste.
"""

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).parent
FEAT_PKL = ROOT / "features.pkl"
MODEL_PKL = ROOT / "model.pkl"
TEST_EXAMS = ROOT / "test_exams.txt"
REPORT_OUT = ROOT / "evaluation_report.txt"

RATERS = ["elaine", "amanda", "marina"]
XGB_BASELINE = {
    "gpd": 0.451,
    "grda": 0.000,
    "lpd": 0.022,
    "lrda": 0.000,
    "normal": 0.749,
    "other": 0.046,
    "seizure": 0.695,
    "macro": 0.280,
}


def predict_all(X, model, le):
    encoded = model.predict(X)
    encoded = np.asarray(encoded, dtype=int)
    return le.inverse_transform(encoded)


def per_class_metrics(y_true, y_pred, classes):
    rows = {}
    for cls in classes:
        y_true_bin = (np.array(y_true) == cls).astype(int)
        y_pred_bin = (np.array(y_pred) == cls).astype(int)
        rows[cls] = {
            "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
            "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
            "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
            "support": int(y_true_bin.sum()),
        }
    return rows


def fmt_table(rows: dict, title: str) -> str:
    lines = [title, "-" * len(title)]
    lines.append(f"  {'class':10s}  {'precision':>9}  {'recall':>7}  {'f1':>7}  {'support':>8}")
    for cls in sorted(rows):
        row = rows[cls]
        lines.append(
            f"  {cls:10s}  {row['precision']:9.3f}  {row['recall']:7.3f}  {row['f1']:7.3f}  {row['support']:8d}"
        )
    macro_f1 = np.mean([rows[cls]["f1"] for cls in sorted(rows)])
    lines.append(f"  {'macro avg':10s}  {'':9s}  {'':7s}  {macro_f1:7.3f}")
    return "\n".join(lines)


def kappa_interp(kappa: float) -> str:
    if kappa < 0:
        return "sem concordância"
    if kappa < 0.20:
        return "leve"
    if kappa < 0.40:
        return "razoável"
    if kappa < 0.60:
        return "moderada"
    if kappa < 0.80:
        return "substancial"
    return "quase perfeita"


def main():
    print("Carregando features e modelo ...")
    with open(FEAT_PKL, "rb") as f:
        bundle = pickle.load(f)
    with open(MODEL_PKL, "rb") as f:
        model_bundle = pickle.load(f)

    X, meta = bundle["X"], bundle["meta"]
    model = model_bundle["model"]
    le = model_bundle["label_encoder"]
    model_type = model_bundle.get("model_type", type(model).__name__)

    test_exam_ids = set(TEST_EXAMS.read_text().splitlines())
    test_idx = [i for i, row in enumerate(meta) if row["exam_id"] in test_exam_ids]
    X_test = X[test_idx]
    meta_test = [meta[i] for i in test_idx]

    y_consensus = np.array([row["label"] for row in meta_test])
    y_ai = predict_all(X_test, model, le)
    y_raters = {
        rater: np.array([row["labels_per_rater"].get(rater, "normal") for row in meta_test])
        for rater in RATERS
    }
    all_classes = sorted(set(le.classes_))

    lines = []
    lines.append("=" * 70)
    lines.append(f"RELATÓRIO DE AVALIAÇÃO — AI vs MÉDICAS  [modelo: {model_type}]")
    lines.append("=" * 70)
    lines.append(f"Janelas avaliadas: {len(test_idx)}")
    lines.append(f"Exames de teste:   {len(test_exam_ids)}")
    lines.append(f"Modelo:            {model_type}")
    lines.append(f"Classes:           {all_classes}")
    lines.append("")

    lines.append("━" * 70)
    lines.append("1. AI vs CONSENSO (ground truth = votação majoritária das 3 médicas)")
    lines.append("━" * 70)
    consensus_rows = per_class_metrics(y_consensus, y_ai, all_classes)
    lines.append(fmt_table(consensus_rows, "AI vs Consenso"))
    f1_ai_consensus = np.mean([consensus_rows[cls]["f1"] for cls in all_classes])
    lines.append(f"\n  F1-macro: {f1_ai_consensus:.4f}")
    lines.append("")

    lines.append("━" * 70)
    lines.append("2. AI vs CADA MÉDICA individualmente")
    lines.append("━" * 70)
    ai_f1_per_rater = {}
    for rater in RATERS:
        rows = per_class_metrics(y_raters[rater], y_ai, all_classes)
        lines.append(fmt_table(rows, f"AI vs {rater.capitalize()}"))
        ai_f1_per_rater[rater] = np.mean([rows[cls]["f1"] for cls in all_classes])
        lines.append(f"\n  F1-macro: {ai_f1_per_rater[rater]:.4f}\n")

    lines.append("━" * 70)
    lines.append("3. INTER-RATER AGREEMENT entre médicas (teto de referência)")
    lines.append("━" * 70)
    kappas = {}
    for r1, r2 in [("elaine", "amanda"), ("elaine", "marina"), ("amanda", "marina")]:
        kappas[(r1, r2)] = cohen_kappa_score(y_raters[r1], y_raters[r2])
        lines.append(f"  {r1.capitalize():8s} vs {r2.capitalize():8s}:  κ = {kappas[(r1, r2)]:.4f}")
    mean_kappa = np.mean(list(kappas.values()))
    lines.append(f"\n  κ médio: {mean_kappa:.4f}")
    lines.append("")

    lines.append("━" * 70)
    lines.append("4. COMPARAÇÃO COM BASELINE HISTÓRICO (XGBoost da rodada anterior)")
    lines.append("━" * 70)
    lines.append("  classe       XGBoost   Atual      delta")
    lines.append("  ------------------------------------------")
    for cls in all_classes:
        prev = XGB_BASELINE.get(cls, 0.0)
        cur = consensus_rows[cls]["f1"]
        delta = cur - prev
        arrow = "↑" if delta > 0.0005 else ("↓" if delta < -0.0005 else "=")
        lines.append(f"  {cls:10s}  {prev:7.3f}  {cur:7.3f}  {delta:+7.3f} {arrow}")
    macro_delta = f1_ai_consensus - XGB_BASELINE["macro"]
    macro_arrow = "↑" if macro_delta > 0.0005 else ("↓" if macro_delta < -0.0005 else "=")
    lines.append("  ------------------------------------------")
    lines.append(f"  {'MACRO':10s}  {XGB_BASELINE['macro']:7.3f}  {f1_ai_consensus:7.3f}  {macro_delta:+7.3f} {macro_arrow}")
    lines.append("")

    lines.append("━" * 70)
    lines.append("5. RESUMO COMPARATIVO")
    lines.append("━" * 70)
    lines.append(f"  AI vs Consenso          F1-macro = {f1_ai_consensus:.4f}")
    for rater in RATERS:
        lines.append(f"  AI vs {rater.capitalize():8s}          F1-macro = {ai_f1_per_rater[rater]:.4f}")
    lines.append("")
    lines.append(f"  Teto de referência (κ médio inter-médicas): {mean_kappa:.4f}")
    lines.append("")
    lines.append("  Interpretação dos kappas:")
    for (r1, r2), kappa in kappas.items():
        lines.append(f"    {r1.capitalize()} vs {r2.capitalize()}: {kappa_interp(kappa)} ({kappa:.4f})")
    lines.append("")

    lines.append("━" * 70)
    lines.append("6. F1-MACRO POR EXAME DE TESTE")
    lines.append("━" * 70)
    exam_windows = defaultdict(list)
    for idx, row in enumerate(meta_test):
        exam_windows[row["exam_id"]].append(idx)

    exam_f1s = []
    for exam_id in sorted(exam_windows):
        idxs = exam_windows[exam_id]
        y_true_exam = y_consensus[idxs]
        y_pred_exam = y_ai[idxs]
        classes_present = sorted(set(y_true_exam))
        exam_f1 = f1_score(
            y_true_exam,
            y_pred_exam,
            average="macro",
            labels=classes_present,
            zero_division=0,
        )
        exam_f1s.append(exam_f1)
        lines.append(
            f"  {exam_id:40s}  F1={exam_f1:.3f}  ({len(idxs)} janelas, classes={classes_present})"
        )
    lines.append(f"\n  F1-macro médio por exame: {np.mean(exam_f1s):.4f} ± {np.std(exam_f1s):.4f}")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)
    REPORT_OUT.write_text(report)
    print(f"\nRelatório salvo em {REPORT_OUT}")


if __name__ == "__main__":
    main()
