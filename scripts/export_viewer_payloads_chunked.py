#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from api_server import DEFAULT_CHUNK_SAMPLES, build_chunked_exam_payload, exam_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Export production-ready viewer payload manifests and chunks from the local EEG pipeline."
    )
    parser.add_argument(
        "--out-dir",
        default="viewer_payload_exports_chunked",
        help="Directory where manifest.json and chunk files will be written.",
    )
    parser.add_argument(
        "--exam-id",
        action="append",
        dest="exam_ids",
        help="Specific exam_id to export. Repeat to export more than one.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of exams to export from the manifest.",
    )
    parser.add_argument(
        "--target-sfreq",
        type=float,
        default=256.0,
        help="Target sampling frequency for the exported viewer payloads.",
    )
    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=DEFAULT_CHUNK_SAMPLES,
        help="Maximum number of downsampled samples per chunk file.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = exam_manifest()
    manifest_by_id = {row["exam_id"]: row for row in manifest}

    if args.exam_ids:
        selected_ids = [exam_id for exam_id in args.exam_ids if exam_id in manifest_by_id]
        missing_ids = sorted(set(args.exam_ids) - set(selected_ids))
        if missing_ids:
            print("Skipping unknown exam ids:", ", ".join(missing_ids))
    else:
        selected_ids = [row["exam_id"] for row in manifest]

    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]

    index = {}

    for idx, exam_id in enumerate(selected_ids, start=1):
        exported = build_chunked_exam_payload(
            exam_id,
            target_sfreq=args.target_sfreq,
            chunk_samples=args.chunk_samples,
        )
        if exported is None:
            print(f"[{idx}/{len(selected_ids)}] skipped {exam_id} (payload unavailable)")
            continue
        exam_dir = out_dir / exam_id
        chunks_dir = exam_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        manifest_payload = exported["manifest"]
        manifest_path = exam_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_payload, separators=(",", ":")), encoding="utf-8")

        largest_chunk_bytes = 0
        total_chunk_bytes = 0
        chunk_sizes = []

        for chunk in exported["chunks"]:
            chunk_path = chunks_dir / f"{chunk['index']:04d}.json"
            chunk_body = {
                "exam_id": exam_id,
                "chunk_index": chunk["index"],
                "start_sample": chunk["start_sample"],
                "end_sample": chunk["end_sample"],
                "samples": chunk["samples"],
                "t0": chunk["t0"],
                "t1": chunk["t1"],
                "rawDs": chunk["rawDs"],
            }
            chunk_path.write_text(json.dumps(chunk_body, separators=(",", ":")), encoding="utf-8")
            chunk_bytes = chunk_path.stat().st_size
            largest_chunk_bytes = max(largest_chunk_bytes, chunk_bytes)
            total_chunk_bytes += chunk_bytes
            chunk_sizes.append(chunk_bytes)

        manifest_bytes = manifest_path.stat().st_size
        total_bytes = manifest_bytes + total_chunk_bytes
        index[exam_id] = {
            "manifest_path": f"{exam_id}/manifest.json",
            "chunks_prefix": f"{exam_id}/chunks",
            "bytes_manifest": manifest_bytes,
            "bytes_chunks_total": total_chunk_bytes,
            "bytes_total": total_bytes,
            "chunk_count": len(exported["chunks"]),
            "largest_chunk_bytes": largest_chunk_bytes,
            "chunk_sizes": chunk_sizes,
            "patient_id": manifest_by_id[exam_id].get("patient_id"),
            "duration_s": manifest_by_id[exam_id].get("duration_s"),
        }
        print(
            f"[{idx}/{len(selected_ids)}] exported {exam_id} -> {manifest_path} "
            f"({len(exported['chunks'])} chunks, max {largest_chunk_bytes:,} bytes, total {total_bytes:,} bytes)"
        )

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote index -> {index_path}")


if __name__ == "__main__":
    main()
