#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from api_server import exam_manifest, exam_payload


def main():
    parser = argparse.ArgumentParser(
        description="Export production-ready viewer payload JSON files from the local EEG pipeline."
    )
    parser.add_argument(
        "--out-dir",
        default="viewer_payload_exports",
        help="Directory where payload JSON files will be written.",
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
        payload = exam_payload(exam_id)
        out_path = out_dir / f"{exam_id}.json"
        out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        byte_size = out_path.stat().st_size
        index[exam_id] = {
            "path": out_path.name,
            "bytes": byte_size,
            "patient_id": manifest_by_id[exam_id].get("patient_id"),
            "duration_s": manifest_by_id[exam_id].get("duration_s"),
        }
        print(f"[{idx}/{len(selected_ids)}] exported {exam_id} -> {out_path} ({byte_size:,} bytes)")

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote index -> {index_path}")


if __name__ == "__main__":
    main()
