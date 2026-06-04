import json
import pickle
import urllib.parse
from collections import Counter, defaultdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
VIEWER_DIR = ROOT / "viewer"
DATASET_PKL = ROOT / "dataset.pkl"
FEATURES_PKL = ROOT / "features.pkl"
MEGA_FEATURES_V1 = ROOT / "mega_features_v1.pkl"
MEGA_MODEL_PKL = ROOT / "model_mega.pkl"
MODEL_PKL = ROOT / "model.pkl"

TARGET_SFREQ = 100.0
RATERS = ["elaine", "amanda", "marina"]
LABEL_COLORS = {
    "normal": "#4CAF50",
    "gpd": "#2196F3",
    "lpd": "#9C27B0",
    "lrda": "#FF9800",
    "grda": "#FF5722",
    "seizure": "#F44336",
    "other": "#9E9E9E",
}


def load_artifacts():
    dataset = pickle.load(open(DATASET_PKL, "rb"))
    features_bundle = pickle.load(open(FEATURES_PKL, "rb"))
    model_path = MEGA_MODEL_PKL if MEGA_MODEL_PKL.exists() else MODEL_PKL
    model_bundle = pickle.load(open(model_path, "rb"))
    return dataset, features_bundle, model_bundle, model_path.name


DATASET, FEATURES_BUNDLE, MODEL_BUNDLE, MODEL_NAME = load_artifacts()
META = FEATURES_BUNDLE["meta"]
MODEL = MODEL_BUNDLE["model"]
LE = MODEL_BUNDLE["label_encoder"]
if MODEL_BUNDLE.get("feature_set") == "v1" and MEGA_FEATURES_V1.exists():
    X = pickle.load(open(MEGA_FEATURES_V1, "rb"))
    FEATURE_MEAN = MODEL_BUNDLE.get("feature_mean")
    FEATURE_STD = MODEL_BUNDLE.get("feature_std")
else:
    X = FEATURES_BUNDLE["X"]
    FEATURE_MEAN = MODEL_BUNDLE.get("mu")
    FEATURE_STD = MODEL_BUNDLE.get("std")
EXAM_WINDOWS = defaultdict(list)
for idx, row in enumerate(DATASET):
    EXAM_WINDOWS[row["exam_id"]].append(idx)


def predict_all_windows():
    X_all = X
    if FEATURE_MEAN is not None and FEATURE_STD is not None:
        X_all = (X_all - np.asarray(FEATURE_MEAN)) / np.asarray(FEATURE_STD)
    if hasattr(MODEL, "predict_proba"):
        proba = np.asarray(MODEL.predict_proba(X_all))
        pred_idx = np.argmax(proba, axis=1)
        conf = np.max(proba, axis=1)
    else:
        pred_idx = np.asarray(MODEL.predict(X_all), dtype=int)
        conf = np.ones(len(pred_idx), dtype=float)
    pred_labels = LE.inverse_transform(pred_idx)
    return pred_labels, conf


ALL_PRED_LABELS, ALL_PRED_CONF = predict_all_windows()


def downsample(signal: np.ndarray, orig_sfreq: float, target_sfreq: float = TARGET_SFREQ):
    factor = max(1, int(round(orig_sfreq / target_sfreq)))
    return signal[:, ::factor], orig_sfreq / factor


def merge_segments(items):
    segments = []
    current = None
    for item in items:
        if current and current["label"] == item["label"]:
            current["t1"] = item["t1"]
            if "confidence" in item:
                current["confidence"] = max(current["confidence"], item["confidence"])
        else:
            current = dict(item)
            segments.append(current)
    return segments


def exam_manifest():
    manifest = []
    for exam_id in sorted(EXAM_WINDOWS):
        rows = [DATASET[i] for i in EXAM_WINDOWS[exam_id]]
        meta_rows = [META[i] for i in EXAM_WINDOWS[exam_id]]
        labels = Counter(row["label"] for row in rows)
        ai_labels = [str(ALL_PRED_LABELS[i]) for i in EXAM_WINDOWS[exam_id]]
        ai_counts = Counter(ai_labels)
        legacy_final_by_rater = {}
        for rater in RATERS:
            rater_counts = Counter(meta_row["labels_per_rater"].get(rater, "normal") for meta_row in meta_rows)
            legacy_final_by_rater[rater] = rater_counts.most_common(1)[0][0] if rater_counts else None
        legacy_consensus_counts = Counter(label for label in legacy_final_by_rater.values() if label)
        legacy_consensus_label = None
        legacy_consensus_count = 0
        if legacy_consensus_counts:
            legacy_consensus_label, legacy_consensus_count = legacy_consensus_counts.most_common(1)[0]
        has_consensus = legacy_consensus_count >= 2
        accuracy_vs_consensus = (
            sum(int(ai_label == row["label"]) for ai_label, row in zip(ai_labels, rows)) / len(rows)
            if rows else None
        )
        manifest.append({
            "exam_id": exam_id,
            "patient_id": rows[0]["patient_id"],
            "duration_s": float(rows[-1]["window_end"]),
            "n_windows": len(rows),
            "dominant_label": labels.most_common(1)[0][0],
            "label_counts": dict(labels),
            "consensus_dominant_label": labels.most_common(1)[0][0],
            "consensus_label_counts": dict(labels),
            "ai_dominant_label": ai_counts.most_common(1)[0][0] if ai_counts else None,
            "ai_label_counts": dict(ai_counts),
            "accuracy_vs_consensus": round(float(accuracy_vs_consensus), 4) if (accuracy_vs_consensus is not None and has_consensus) else None,
            "legacy_final_by_rater": legacy_final_by_rater,
            "consensus_available": has_consensus,
            "consensus_dominant_label_final": legacy_consensus_label if has_consensus else None,
        })
    return manifest


MANIFEST = exam_manifest()


def exam_payload(exam_id: str):
    idxs = EXAM_WINDOWS.get(exam_id)
    if not idxs:
        return None
    rows = [DATASET[i] for i in idxs]
    meta_rows = [META[i] for i in idxs]
    signal = np.concatenate([row["signal"] for row in rows], axis=1)
    sfreq = float(rows[0]["sfreq"])
    signal_ds, ds_freq = downsample(signal, sfreq)

    X_exam = X[idxs]
    if FEATURE_MEAN is not None and FEATURE_STD is not None:
        X_exam = (X_exam - np.asarray(FEATURE_MEAN)) / np.asarray(FEATURE_STD)
    if hasattr(MODEL, "predict_proba"):
        proba = np.asarray(MODEL.predict_proba(X_exam))
        pred_idx = np.argmax(proba, axis=1)
        conf = np.max(proba, axis=1)
    else:
        pred_idx = np.asarray(MODEL.predict(X_exam), dtype=int)
        conf = np.ones(len(pred_idx), dtype=float)
    pred_labels = LE.inverse_transform(pred_idx)

    predictions = []
    ai_track_items = []
    for meta_row, label, prob in zip(meta_rows, pred_labels, conf):
        item = {
            "window": int(meta_row["window_idx"]),
            "t0": float(meta_row["window_start"]),
            "t1": float(meta_row["window_end"]),
            "label": str(label),
            "confidence": float(prob),
        }
        predictions.append(item)
        ai_track_items.append(item)

    tracks = {"AI": merge_segments(ai_track_items)}
    for rater in RATERS:
        track_items = [
            {
                "t0": float(meta_row["window_start"]),
                "t1": float(meta_row["window_end"]),
                "label": str(meta_row["labels_per_rater"].get(rater, "normal")),
            }
            for meta_row in meta_rows
        ]
        tracks[rater.capitalize()] = merge_segments(track_items)

    return {
        "exam_id": exam_id,
        "model_name": MODEL_NAME,
        "channels": rows[0]["ch_names"],
        "duration": float(rows[-1]["window_end"]),
        "sfreq": sfreq,
        "dsFreq": ds_freq,
        "rawDs": signal_ds.tolist(),
        "predictions": predictions,
        "tracks": tracks,
        "consensus_counts": dict(Counter(row["label"] for row in rows)),
        "label_colors": LABEL_COLORS,
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/exams":
            self.respond_json({"exams": MANIFEST})
            return
        if parsed.path == "/api/exam":
            exam_id = urllib.parse.parse_qs(parsed.query).get("id", [None])[0]
            payload = exam_payload(exam_id) if exam_id else None
            if payload is None:
                self.send_error(404, "Exam not found")
                return
            self.respond_json(payload)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def respond_json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8123), Handler)
    print("Serving on http://127.0.0.1:8123")
    server.serve_forever()


if __name__ == "__main__":
    main()
