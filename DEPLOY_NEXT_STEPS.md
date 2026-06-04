# Neurogram AI Training — GitHub and Deploy Next Steps

This project is in a strong product-prototype state, but it is **not production-ready yet** because part of the app still depends on local files and a local Python server.

## What is ready to go into GitHub now

These should be versioned:

- `viewer/index.html`
- `api_server.py`
- `ingest.py`
- `features.py`
- `train.py`
- `evaluate.py`
- `mega_train.py`
- `agreement_train.py`
- `visualizer.py`
- `exam_mapping.csv`
- `viewer/legacy_final_reviews.json`
- viewer helper HTML files if still useful for internal workflows

These are product code / configuration and are safe to keep in source control.

## What should NOT go into GitHub

These are local-only, too large, sensitive, or derived:

- `data/`
- `data/exams/`
- all `.edf` files
- all `.pkl` files
- temporary JSON seeds like `tmp_*.json`
- temp SQL chunk folders
- ad hoc result reports that can be regenerated

## Current architecture

### Already production-friendly

- Frontend app inside `viewer/index.html`
- Supabase Auth
- Supabase database-backed exam list
- Supabase-backed user profiles

### Still local-only

- `/api/exams` and `/api/exam` are currently served by `api_server.py`
- EEG signals are still read from local files
- exam-derived local manifest / AI payload flow still assumes local disk access

## What is already prepared for Netlify

- `netlify.toml` at the repository root
- `viewer/` prepared as the publish directory
- `netlify/functions/runtime-config.mts` for runtime Supabase config
- `netlify/functions/exams.mts` as the first production-ready exam manifest endpoint
- `netlify/functions/exam.mts` as an explicit placeholder for the remaining exam payload migration
- `.env.example` with the expected public environment variable names

## What must change before full Netlify deployment

### 1. Set Netlify environment variables

In Netlify project settings, add:

- `SUPABASE_URL`
- `SUPABASE_PUBLISHABLE_KEY`

The frontend now tries to read them from `/api/runtime-config` first and only falls back to the current local values if unavailable.

### 2. Replace the local API server

The following endpoints need to become real production endpoints:

- `GET /api/exams`
- `GET /api/exam?id=...`

Current status:

- `/api/exams` now has an initial Netlify Function implementation backed by Supabase + legacy review fallback data
- `/api/exam` still needs full production implementation

Best target:

- Netlify Functions

Alternative:

- another hosted API service

### 3. Move exam source data off the laptop

Production cannot depend on local EDFs. We need a hosted source for:

- uploaded EDF files
- precomputed per-exam outputs
- legacy review summaries

Most likely home:

- Supabase Storage

### 3.1. Viewer payload strategy

The current viewer expects a rich `/api/exam` payload with:

- downsampled EEG signal arrays (`rawDs`)
- predictions by 10-second window
- annotation tracks per reviewer
- metadata such as channels, duration, and sampling rate

Measured locally, these payloads can be **very large** (tens to hundreds of MB per exam), so they should **not** be stored directly in Postgres rows.

Recommended production path:

1. Export one JSON payload per exam from the local pipeline.
2. Upload those JSON payloads to hosted storage.
3. Save either:
   - `exams.metadata.viewer_payload_storage_path`
   - `exams.metadata.viewer_payload_url`
   - `exams.metadata.viewer_payload_public_url`
   - `exams.metadata.viewer_payload_path`
   - or the equivalent fields inside `exam_ai_reviews.summary`
4. Let `GET /api/exam` proxy or fetch the payload from there.

There is now a local export helper:

- `scripts/export_viewer_payloads.py`
- `scripts/build_viewer_payload_backfill.py`

Example:

```bash
python3 scripts/export_viewer_payloads.py --limit 5
python3 scripts/export_viewer_payloads.py --exam-id PAT-24P6-EHJN-5ARW_0
python3 scripts/build_viewer_payload_backfill.py --field viewer_payload_storage_path --base eeg-viewer-payloads
python3 scripts/build_viewer_payload_backfill.py --mode url --base https://<public-host>/payloads
```

These outputs are intentionally gitignored via:

- `viewer_payload_exports/`

Recommended metadata contract for production:

- **Preferred (safer):** `exams.metadata.viewer_payload_storage_path`
  - for private storage object paths such as:
    - `eeg-viewer-payloads/PAT-24P6-EHJN-5ARW_0.json`
- `exams.metadata.viewer_payload_path`
  - for public storage-style paths
- or `exams.metadata.viewer_payload_url`
  - for full public URLs

The current production `GET /api/exam` already supports:

- `exams.metadata.viewer_payload_storage_path`
- `exams.metadata.viewer_payload_url`
- `exams.metadata.viewer_payload_public_url`
- `exams.metadata.viewer_payload_path`
- `exam_ai_reviews.summary.viewer_payload_storage_path`
- the same three public keys inside `exam_ai_reviews.summary`

Suggested first hosted layout:

- bucket / prefix: `eeg-viewer-payloads/`
- file naming: `<exam_code>.json`

Recommended first rollout:

1. Create a **private** storage bucket named `eeg-viewer-payloads`
2. Upload the exported payload JSON files there
3. Backfill `viewer_payload_storage_path`
4. Let `/api/exam` read them via the server-side function

Then run a backfill using the generated SQL artifact and the endpoint will start resolving those exams immediately.

### 4. Decide the production inference strategy

There are two valid directions:

#### Option A — precompute on upload

When a new exam is uploaded:

- store EDF
- run AI pipeline once
- save exam-level and window-level outputs
- viewer reads precomputed payload

This is the recommended first production path.

#### Option B — compute on demand

When a user opens an exam:

- fetch EDF
- run inference live

This is heavier, slower, and more operationally complex.

## Recommended release path

### Phase 1 — GitHub snapshot

- initialize local Git repo
- commit product frontend and supporting scripts
- push to `rodmag-ngram/ngram-ai-training`

### Phase 2 — deploy-safe frontend

- add environment handling for Supabase URL / anon key
- connect Netlify build and publish directory
- publish static frontend

### Phase 3 — production API

- port local API behavior to Netlify Functions
- return the same JSON shape the viewer already expects

### Phase 4 — storage / ingestion

- upload EDFs to Supabase Storage
- persist AI outputs per exam
- remove dependency on local files

## Practical recommendation right now

The next best move is:

1. initialize Git locally
2. commit the current UI / Supabase work
3. push to GitHub
4. then convert `api_server.py` into production endpoints

That gives us a safe checkpoint before the bigger backend migration.
