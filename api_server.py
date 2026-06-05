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

TARGET_SFREQ = 256.0
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
DEFAULT_CHUNK_SAMPLES = 40_000


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
    if factor <= 1:
        return signal, orig_sfreq

    usable = signal.shape[1] - (signal.shape[1] % factor)
    if usable <= 0:
        return signal[:, ::factor], orig_sfreq / factor

    trimmed = signal[:, :usable]
    reshaped = trimmed.reshape(signal.shape[0], -1, factor)
    mean = reshaped.mean(axis=2, keepdims=True)
    offsets = np.abs(reshaped - mean)
    peak_idx = offsets.argmax(axis=2)
    selected = np.take_along_axis(reshaped, peak_idx[..., None], axis=2).squeeze(axis=2)
    return selected, orig_sfreq / factor


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
        per_label_stats = {}
        matched_window_count = 0
        consensus_window_count = 0
        for ai_label, meta_row in zip(ai_labels, meta_rows):
            reviewer_counts = Counter(
                str(meta_row["labels_per_rater"].get(rater, "normal"))
                for rater in RATERS
            )
            consensus_label = None
            consensus_count = 0
            if reviewer_counts:
                consensus_label, consensus_count = reviewer_counts.most_common(1)[0]
            if not consensus_label or consensus_count < 2:
                continue

            consensus_window_count += 1
            label_stats = per_label_stats.setdefault(consensus_label, {
                "consensus_window_count": 0,
                "matched_window_count": 0,
            })
            label_stats["consensus_window_count"] += 1
            if ai_label == consensus_label:
                label_stats["matched_window_count"] += 1
                matched_window_count += 1
        accuracy_vs_consensus = (
            matched_window_count / consensus_window_count
            if consensus_window_count else None
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
            "accuracy_vs_consensus": round(float(accuracy_vs_consensus), 4) if accuracy_vs_consensus is not None else None,
            "legacy_final_by_rater": legacy_final_by_rater,
            "consensus_available": has_consensus,
            "consensus_dominant_label_final": legacy_consensus_label if has_consensus else None,
            "window_metrics": {
                "consensus_window_count": consensus_window_count,
                "matched_window_count": matched_window_count,
                "window_agreement": round(float(accuracy_vs_consensus), 4) if accuracy_vs_consensus is not None else None,
                "per_label_stats": {
                    label: {
                        **stats,
                        "window_agreement": round(stats["matched_window_count"] / stats["consensus_window_count"], 4) if stats["consensus_window_count"] else None,
                    }
                    for label, stats in per_label_stats.items()
                },
            },
        })
    return manifest


MANIFEST = exam_manifest()


def exam_payload(exam_id: str, target_sfreq: float = TARGET_SFREQ):
    idxs = EXAM_WINDOWS.get(exam_id)
    if not idxs:
        return None
    rows = [DATASET[i] for i in idxs]
    meta_rows = [META[i] for i in idxs]
    signal = np.concatenate([row["signal"] for row in rows], axis=1)
    sfreq = float(rows[0]["sfreq"])
    signal_ds, ds_freq = downsample(signal, sfreq, target_sfreq=target_sfreq)

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


def chunk_name(index: int) -> str:
    return f"{index:04d}.json"


def build_chunked_exam_payload(exam_id: str, target_sfreq: float = TARGET_SFREQ, chunk_samples: int = DEFAULT_CHUNK_SAMPLES):
    payload = exam_payload(exam_id, target_sfreq=target_sfreq)
    if payload is None:
        return None

    raw_ds = payload.pop("rawDs")
    total_samples = len(raw_ds[0]) if raw_ds else 0
    chunk_samples = max(1, int(chunk_samples))
    raw_chunks = []

    for chunk_index, start in enumerate(range(0, total_samples, chunk_samples)):
        end = min(total_samples, start + chunk_samples)
        chunk_raw = [channel[start:end] for channel in raw_ds]
        raw_chunks.append({
            "index": chunk_index,
            "path": f"chunks/{chunk_name(chunk_index)}",
            "start_sample": start,
            "end_sample": end,
            "samples": end - start,
            "t0": start / payload["dsFreq"] if payload["dsFreq"] else 0.0,
            "t1": end / payload["dsFreq"] if payload["dsFreq"] else 0.0,
            "rawDs": chunk_raw,
        })

    manifest = {
        **payload,
        "format": "chunked-v1",
        "chunked": True,
        "sample_count": total_samples,
        "chunk_samples": chunk_samples,
        "chunks": [
            {key: value for key, value in chunk.items() if key != "rawDs"}
            for chunk in raw_chunks
        ],
    }
    return {"manifest": manifest, "chunks": raw_chunks}


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
            exported = build_chunked_exam_payload(exam_id) if exam_id else None
            payload = exported["manifest"] if exported else None
            if payload is None:
                self.send_error(404, "Exam not found")
                return
            self.respond_json(payload)
            return
        if parsed.path == "/api/exam-full":
            exam_id = urllib.parse.parse_qs(parsed.query).get("id", [None])[0]
            payload = exam_payload(exam_id) if exam_id else None
            if payload is None:
                self.send_error(404, "Exam not found")
                return
            self.respond_json(payload)
            return
        if parsed.path == "/api/exam-manifest":
            query = urllib.parse.parse_qs(parsed.query)
            exam_id = query.get("id", [None])[0]
            chunk_samples = int(query.get("chunk_samples", [DEFAULT_CHUNK_SAMPLES])[0])
            payload = build_chunked_exam_payload(exam_id, chunk_samples=chunk_samples) if exam_id else None
            if payload is None:
                self.send_error(404, "Exam not found")
                return
            self.respond_json(payload["manifest"])
            return
        if parsed.path == "/api/exam-chunk":
            query = urllib.parse.parse_qs(parsed.query)
            exam_id = query.get("id", [None])[0]
            chunk_index_raw = query.get("chunk", [None])[0]
            chunk_samples = int(query.get("chunk_samples", [DEFAULT_CHUNK_SAMPLES])[0])
            if exam_id is None or chunk_index_raw is None:
                self.send_error(400, "Missing exam id or chunk index")
                return
            payload = build_chunked_exam_payload(exam_id, chunk_samples=chunk_samples)
            if payload is None:
                self.send_error(404, "Exam not found")
                return
            try:
                chunk_index = int(str(chunk_index_raw).replace(".json", ""))
                chunk = payload["chunks"][chunk_index]
            except (ValueError, IndexError):
                self.send_error(404, "Chunk not found")
                return
            self.respond_json(chunk)
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
