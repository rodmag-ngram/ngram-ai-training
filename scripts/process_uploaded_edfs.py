#!/usr/bin/env python3
"""
Process uploaded EDF exams from Supabase and publish chunked viewer payloads.

This worker is meant to run server-side (outside the browser) with access to:
  - local model artifacts (model_mega.pkl / model.pkl, features.pkl)
  - Python scientific stack already used by this repo (mne, numpy, scipy)
  - Supabase service role credentials

Examples:
  python3 scripts/process_uploaded_edfs.py --exam-code MY-UPLOADED-EXAM
  python3 scripts/process_uploaded_edfs.py --once
  python3 scripts/process_uploaded_edfs.py --loop --interval 30
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import mne
import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import CHANNELS, extract_window  # noqa: E402

FEATURES_PKL = ROOT / "features.pkl"
MEGA_MODEL_PKL = ROOT / "model_mega.pkl"
MODEL_PKL = ROOT / "model.pkl"

TARGET_SFREQ = 256.0
DEFAULT_CHUNK_SAMPLES = 40_000
WINDOW_SECONDS = 10.0
LABEL_COLORS = {
    "normal": "#4CAF50",
    "gpd": "#2196F3",
    "lpd": "#9C27B0",
    "lrda": "#FF9800",
    "grda": "#FF5722",
    "seizure": "#F44336",
    "other": "#9E9E9E",
}
CHANNEL_ALIASES = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
    "A1": "CA1",
    "A2": "CA2",
    "EKG": "CA1",
    "ECG": "CA1",
}


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


SUPABASE_URL = require_env("SUPABASE_URL").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = require_env("SUPABASE_SERVICE_ROLE_KEY")
VIEWER_BUCKET = (os.getenv("SUPABASE_VIEWER_PAYLOAD_BUCKET") or "eeg-viewer-payloads").strip("/")
RAW_BUCKET = (os.getenv("SUPABASE_RAW_EDF_BUCKET") or "eeg-raw-edf").strip("/")


def load_model_artifacts():
    features_bundle = pickle.load(open(FEATURES_PKL, "rb"))
    model_path = MEGA_MODEL_PKL if MEGA_MODEL_PKL.exists() else MODEL_PKL
    model_bundle = pickle.load(open(model_path, "rb"))
    model = model_bundle["model"]
    label_encoder = model_bundle["label_encoder"]
    feature_set = model_bundle.get("feature_set") or "v2"
    mean = model_bundle.get("feature_mean")
    std = model_bundle.get("feature_std")
    if mean is None or std is None:
        mean = model_bundle.get("mu")
        std = model_bundle.get("std")
    if mean is None or std is None:
        mean = features_bundle.get("mu")
        std = features_bundle.get("std")
    if mean is None or std is None:
        raise RuntimeError("Could not find feature normalization statistics.")
    return model, label_encoder, np.asarray(mean), np.asarray(std), model_path.name, feature_set


MODEL, LABEL_ENCODER, FEATURE_MEAN, FEATURE_STD, MODEL_NAME, FEATURE_SET = load_model_artifacts()


def supabase_headers(content_type: str | None = None) -> dict[str, str]:
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Accept": "application/json",
    }
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def rest_url(path: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/{path.lstrip('/')}"


def storage_upload_url(bucket: str, path: str) -> str:
    clean = path.lstrip("/")
    return f"{SUPABASE_URL}/storage/v1/object/{bucket}/{clean}"


def storage_download_url(bucket: str, path: str) -> str:
    clean = path.lstrip("/")
    return f"{SUPABASE_URL}/storage/v1/object/authenticated/{bucket}/{clean}"


def request_json(method: str, url: str, payload: Any | None = None, extra_headers: dict[str, str] | None = None):
    data = None
    headers = supabase_headers("application/json; charset=utf-8")
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {detail}") from exc


def request_bytes(method: str, url: str, payload: bytes | None = None, content_type: str = "application/octet-stream", extra_headers: dict[str, str] | None = None):
    headers = supabase_headers(content_type)
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {detail}") from exc


def normalize_channel_name(name: str) -> str:
    token = "".join(ch for ch in str(name or "").upper() if ch.isalnum())
    return CHANNEL_ALIASES.get(token, token)


def align_signal_channels(raw: mne.io.BaseRaw) -> tuple[np.ndarray, list[str], float]:
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])
    index_by_name: dict[str, int] = {}
    for index, name in enumerate(raw.ch_names):
        normalized = normalize_channel_name(name)
        index_by_name.setdefault(normalized, index)

    aligned = []
    for channel in CHANNELS:
        source_index = index_by_name.get(channel)
        if source_index is None:
            aligned.append(np.zeros(data.shape[1], dtype=np.float32))
        else:
            aligned.append(data[source_index].astype(np.float32))
    return np.vstack(aligned), list(CHANNELS), sfreq


def downsample_peak_preserving(signal: np.ndarray, orig_sfreq: float, target_sfreq: float = TARGET_SFREQ):
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


def merge_segments(items: list[dict[str, Any]]):
    segments: list[dict[str, Any]] = []
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


def extract_window_v1(signal: np.ndarray, sfreq: float) -> np.ndarray:
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 70.0),
    }
    band_order = ["delta", "theta", "alpha", "beta", "gamma"]
    feats = []
    for ch in range(signal.shape[0]):
        x = signal[ch].astype(np.float64)
        freqs, psd = welch(x, fs=sfreq, nperseg=256, window="hann")
        total = float(np.sum(psd)) + 1e-12
        band_feats = []
        for band in band_order:
            low, high = bands[band]
            mask = (freqs >= low) & (freqs <= high)
            value = float(np.mean(psd[mask])) if mask.any() else 0.0
            band_feats.append(value / total)
        variance = float(np.var(x))
        zcr = float(np.sum(np.diff(np.sign(x)) != 0)) / len(x)
        feats.extend(band_feats + [variance, zcr])
    return np.array(feats, dtype=np.float32)


def feature_vector_for_window(signal: np.ndarray, sfreq: float) -> np.ndarray:
    if FEATURE_SET == "v1":
        return extract_window_v1(signal, sfreq)
    return extract_window(signal, sfreq)


def chunk_name(index: int) -> str:
    return f"{index:04d}.json"


def build_payload_from_edf(local_path: Path, exam_code: str, patient_code: str | None, chunk_samples: int):
    raw = mne.io.read_raw_edf(str(local_path), preload=True, verbose=False)
    signal, channels, sfreq = align_signal_channels(raw)
    duration = float(raw.times[-1]) if len(raw.times) else 0.0
    n_windows = int(duration // WINDOW_SECONDS)
    if n_windows <= 0:
        raise RuntimeError("EDF is shorter than one full 10-second window.")

    predictions = []
    ai_track_items = []
    label_counts: Counter[str] = Counter()

    for window_idx in range(n_windows):
        t0 = window_idx * WINDOW_SECONDS
        t1 = t0 + WINDOW_SECONDS
        start_idx = int(t0 * sfreq)
        end_idx = int(t1 * sfreq)
        window_signal = signal[:, start_idx:end_idx]
        feature_vector = feature_vector_for_window(window_signal, sfreq)
        standardized = (feature_vector - FEATURE_MEAN) / FEATURE_STD
        if hasattr(MODEL, "predict_proba"):
            proba = np.asarray(MODEL.predict_proba(standardized[None, :]))[0]
            pred_idx = int(np.argmax(proba))
            confidence = float(np.max(proba))
        else:
            pred_idx = int(np.asarray(MODEL.predict(standardized[None, :]), dtype=int)[0])
            confidence = 1.0
        label = str(LABEL_ENCODER.inverse_transform([pred_idx])[0])
        label_counts[label] += 1
        item = {
            "window": window_idx,
            "t0": float(t0),
            "t1": float(t1),
            "label": label,
            "confidence": confidence,
        }
        predictions.append(item)
        ai_track_items.append(item)

    signal_ds, ds_freq = downsample_peak_preserving(signal, sfreq, target_sfreq=TARGET_SFREQ)
    raw_ds = signal_ds.tolist()
    total_samples = len(raw_ds[0]) if raw_ds else 0
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
            "t0": start / ds_freq if ds_freq else 0.0,
            "t1": end / ds_freq if ds_freq else 0.0,
            "rawDs": chunk_raw,
        })

    dominant_label = label_counts.most_common(1)[0][0] if label_counts else "normal"
    manifest = {
        "exam_id": exam_code,
        "patient_id": patient_code or exam_code,
        "model_name": MODEL_NAME,
        "channels": channels,
        "duration": float(n_windows * WINDOW_SECONDS),
        "sfreq": sfreq,
        "dsFreq": ds_freq,
        "predictions": predictions,
        "tracks": {"AI": merge_segments(ai_track_items)},
        "consensus_counts": {},
        "accuracy_vs_consensus": None,
        "window_metrics": {
            "consensus_window_count": 0,
            "matched_window_count": 0,
            "window_agreement": None,
            "per_label_stats": {},
        },
        "label_counts": dict(label_counts),
        "dominant_label": dominant_label,
        "consensus_dominant_label": None,
        "ai_dominant_label": dominant_label,
        "ai_label_counts": dict(label_counts),
        "consensus_available": False,
        "consensus_dominant_label_final": None,
        "label_colors": LABEL_COLORS,
        "format": "chunked-v1",
        "chunked": True,
        "sample_count": total_samples,
        "chunk_samples": chunk_samples,
        "chunks": [{key: value for key, value in chunk.items() if key != "rawDs"} for chunk in raw_chunks],
    }
    return manifest, raw_chunks


def fetch_json_list(path: str):
    payload = request_json("GET", rest_url(path))
    return payload or []


def update_exam_state(exam_id: str, patch: dict[str, Any]):
    path = f"exams?id=eq.{urllib.parse.quote(exam_id)}"
    request_json(
        "PATCH",
        rest_url(path),
        patch,
        extra_headers={"Prefer": "return=minimal"},
    )


def update_ai_review(exam_id: str, patch: dict[str, Any]):
    path = f"exam_ai_reviews?exam_id=eq.{urllib.parse.quote(exam_id)}"
    request_json(
        "PATCH",
        rest_url(path),
        patch,
        extra_headers={"Prefer": "return=minimal"},
    )


def upload_json_object(bucket: str, path: str, payload: Any):
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    request_bytes(
        "POST",
        storage_upload_url(bucket, path),
        payload=body,
        content_type="application/json; charset=utf-8",
        extra_headers={"x-upsert": "true"},
    )


def download_storage_object(bucket: str, path: str) -> bytes:
    return request_bytes("GET", storage_download_url(bucket, path))


def list_processing_exams(limit: int):
    query = (
        "exams?"
        "select=id,exam_code,patient_code,source_file_name,edf_storage_path,status,metadata"
        f"&status=eq.processing_ai&limit={limit}"
    )
    return fetch_json_list(query)


def get_single_exam(exam_code: str | None, exam_id: str | None):
    if exam_id:
        query = (
            "exams?"
            "select=id,exam_code,patient_code,source_file_name,edf_storage_path,status,metadata"
            f"&id=eq.{urllib.parse.quote(exam_id)}&limit=1"
        )
    elif exam_code:
        query = (
            "exams?"
            "select=id,exam_code,patient_code,source_file_name,edf_storage_path,status,metadata"
            f"&exam_code=eq.{urllib.parse.quote(exam_code)}&limit=1"
        )
    else:
        return None
    rows = fetch_json_list(query)
    return rows[0] if rows else None


def ensure_ai_review_row(exam_id: str, raw_edf_storage_path: str):
    existing = fetch_json_list(
        "exam_ai_reviews?"
        "select=exam_id,review_status,summary"
        f"&exam_id=eq.{urllib.parse.quote(exam_id)}&limit=1"
    )
    if existing:
        return
    request_json(
        "POST",
        rest_url("exam_ai_reviews"),
        [{
            "exam_id": exam_id,
            "model_name": "LightGBM",
            "model_version": "v1",
            "pipeline_version": "chunked-upload",
            "review_status": "pending",
            "summary": {
                "ingest_state": "queued",
                "raw_edf_storage_path": raw_edf_storage_path,
            },
        }],
        extra_headers={"Prefer": "return=minimal"},
    )


def process_exam(exam_row: dict[str, Any], chunk_samples: int):
    exam_id = exam_row["id"]
    exam_code = exam_row["exam_code"]
    patient_code = exam_row.get("patient_code") or exam_code
    edf_storage_path = exam_row.get("edf_storage_path")
    if not edf_storage_path:
        raise RuntimeError(f"Exam {exam_code} is missing edf_storage_path.")

    ensure_ai_review_row(exam_id, edf_storage_path)

    metadata = dict(exam_row.get("metadata") or {})
    metadata.update({
        "ingest_state": "running",
        "processing_started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    update_exam_state(exam_id, {"metadata": metadata, "status": "processing_ai"})
    update_ai_review(exam_id, {
        "review_status": "running",
        "summary": {
            "ingest_state": "running",
            "raw_edf_storage_path": edf_storage_path,
        },
    })

    payload_prefix = f"exams/{exam_code}"
    manifest_path = f"{payload_prefix}/manifest.json"

    with tempfile.TemporaryDirectory(prefix="edf-upload-") as tmp_dir:
        tmp_path = Path(tmp_dir) / "source.edf"
        tmp_path.write_bytes(download_storage_object(RAW_BUCKET, edf_storage_path))
        manifest, chunks = build_payload_from_edf(tmp_path, exam_code, patient_code, chunk_samples=chunk_samples)

    upload_json_object(VIEWER_BUCKET, manifest_path, manifest)
    for chunk in chunks:
        upload_json_object(VIEWER_BUCKET, f"{payload_prefix}/{chunk['path']}", chunk)

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata.update({
        "ingest_state": "ready",
        "viewer_payload_storage_path": manifest_path,
        "processed_at": finished_at,
        "chunk_count": len(chunks),
        "sample_count": manifest["sample_count"],
        "chunk_samples": manifest["chunk_samples"],
    })
    summary = {
        "ingest_state": "ready",
        "raw_edf_storage_path": edf_storage_path,
        "viewer_payload_storage_path": manifest_path,
        "processed_at": finished_at,
        "duration": manifest["duration"],
        "window_count": len(manifest["predictions"]),
        "chunk_count": len(chunks),
        "model_name": MODEL_NAME,
        "window_metrics": manifest["window_metrics"],
        "ai_dominant_label": manifest["ai_dominant_label"],
        "label_counts": manifest["label_counts"],
    }
    update_exam_state(exam_id, {
        "status": "ready",
        "duration_seconds": manifest["duration"],
        "metadata": metadata,
    })
    update_ai_review(exam_id, {
        "review_status": "completed",
        "model_name": "LightGBM",
        "model_version": "v1",
        "pipeline_version": "chunked-upload",
        "summary": summary,
        "predictions": manifest["predictions"],
    })
    print(f"[ready] {exam_code} -> {manifest_path} ({len(chunks)} chunks)")


def mark_failed(exam_row: dict[str, Any], error_message: str):
    exam_id = exam_row["id"]
    metadata = dict(exam_row.get("metadata") or {})
    metadata.update({
        "ingest_state": "failed",
        "processing_failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processing_error": error_message[:1000],
    })
    update_exam_state(exam_id, {"status": "failed", "metadata": metadata})
    update_ai_review(exam_id, {
        "review_status": "failed",
        "summary": {
            "ingest_state": "failed",
            "error_message": error_message[:1000],
            "raw_edf_storage_path": exam_row.get("edf_storage_path"),
        },
    })


def process_batch(args):
    processed = 0
    if args.exam_code or args.exam_id:
        exam = get_single_exam(args.exam_code, args.exam_id)
        if not exam:
            print("No exam found for the requested identifier.")
            return 0
        exams = [exam]
    else:
        exams = list_processing_exams(args.limit)

    for exam in exams:
        try:
            process_exam(exam, chunk_samples=args.chunk_samples)
            processed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[failed] {exam.get('exam_code')} -> {exc}")
            try:
                mark_failed(exam, str(exc))
            except Exception as mark_exc:  # noqa: BLE001
                print(f"[warn] could not mark failure for {exam.get('exam_code')}: {mark_exc}")
    return processed


def build_parser():
    parser = argparse.ArgumentParser(description="Process uploaded EDF exams from Supabase.")
    parser.add_argument("--exam-code", help="Process a single exam by exam_code.")
    parser.add_argument("--exam-id", help="Process a single exam by database UUID.")
    parser.add_argument("--once", action="store_true", help="Process one batch and exit.")
    parser.add_argument("--loop", action="store_true", help="Continuously poll for new processing_ai exams.")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds for --loop.")
    parser.add_argument("--limit", type=int, default=10, help="Max exams per batch when polling.")
    parser.add_argument("--chunk-samples", type=int, default=DEFAULT_CHUNK_SAMPLES, help="Samples per payload chunk.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.loop:
        while True:
            count = process_batch(args)
            if count == 0:
                print("[idle] no processing_ai exams found")
            time.sleep(max(1, args.interval))
        return

    process_batch(args)


if __name__ == "__main__":
    main()
