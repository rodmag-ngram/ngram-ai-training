#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def build_payload_reference(base: str, filename: str, mode: str) -> str:
    base = base.rstrip("/")
    if mode == "url":
        return f"{base}/{filename}"
    return f"{base}/{filename}".lstrip("/")


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser(
        description="Build JSON and SQL backfill artifacts for hosted EEG viewer payloads."
    )
    parser.add_argument(
        "--index",
        default="viewer_payload_exports/index.json",
        help="Path to the export index.json created by export_viewer_payloads.py",
    )
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Treat the export index as chunked and point metadata to manifest.json paths.",
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Base public URL or storage path prefix where the payload JSON files will live.",
    )
    parser.add_argument(
        "--mode",
        choices=["url", "path"],
        default="path",
        help="Whether to write viewer_payload_url or viewer_payload_path references.",
    )
    parser.add_argument(
        "--field",
        choices=["viewer_payload_url", "viewer_payload_path", "viewer_payload_public_url", "viewer_payload_storage_path"],
        default=None,
        help="Override the metadata field name. Defaults to viewer_payload_url for mode=url and viewer_payload_path for mode=path.",
    )
    parser.add_argument(
        "--json-out",
        default="viewer_payload_exports/backfill_payload_refs.json",
        help="Where to write the JSON mapping artifact.",
    )
    parser.add_argument(
        "--sql-out",
        default="viewer_payload_exports/backfill_payload_refs.sql",
        help="Where to write the SQL update artifact.",
    )
    args = parser.parse_args()

    field_name = args.field or ("viewer_payload_url" if args.mode == "url" else "viewer_payload_path")

    index_path = Path(args.index)
    if not index_path.exists():
        raise SystemExit(f"Index file not found: {index_path}")

    with index_path.open("r", encoding="utf-8") as fh:
        index = json.load(fh)

    mapping = {}
    sql_lines = [
        "-- Backfill viewer payload references into public.exams.metadata",
        "-- Review before applying.",
        "",
    ]

    for exam_id, item in sorted(index.items()):
        if args.chunked:
            filename = item.get("manifest_path") or item.get("path") or f"{exam_id}/manifest.json"
        else:
            filename = item["path"]
        ref = build_payload_reference(args.base, filename, args.mode)
        mapping[exam_id] = {
            field_name: ref,
            "bytes": item.get("bytes_total", item.get("bytes")),
            "patient_id": item.get("patient_id"),
            "duration_s": item.get("duration_s"),
        }
        if args.chunked:
            mapping[exam_id]["chunk_count"] = item.get("chunk_count")
            mapping[exam_id]["largest_chunk_bytes"] = item.get("largest_chunk_bytes")
        sql_lines.append(
            "update public.exams "
            f"set metadata = coalesce(metadata, '{{}}'::jsonb) || jsonb_build_object({sql_literal(field_name)}, {sql_literal(ref)}) "
            f"where exam_code = {sql_literal(exam_id)};"
        )

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    sql_out = Path(args.sql_out)
    sql_out.parent.mkdir(parents=True, exist_ok=True)
    sql_out.write_text("\n".join(sql_lines) + "\n", encoding="utf-8")

    print(f"Wrote JSON mapping -> {json_out}")
    print(f"Wrote SQL backfill -> {sql_out}")
    print(f"Field name -> {field_name}")
    print(f"Entries -> {len(mapping)}")


if __name__ == "__main__":
    main()
