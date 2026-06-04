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

## What must change before full Netlify deployment

### 1. Replace the local API server

The following endpoints need to become real production endpoints:

- `GET /api/exams`
- `GET /api/exam?id=...`

Best target:

- Netlify Functions

Alternative:

- another hosted API service

### 2. Move exam source data off the laptop

Production cannot depend on local EDFs. We need a hosted source for:

- uploaded EDF files
- precomputed per-exam outputs
- legacy review summaries

Most likely home:

- Supabase Storage

### 3. Decide the production inference strategy

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
