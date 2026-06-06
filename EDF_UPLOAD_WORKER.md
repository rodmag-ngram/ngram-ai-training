# EDF Upload Worker

This worker closes the server-side ingestion loop for `Upload Exam`.

## What it does

When an admin uploads a raw EDF in the app:

1. the file is stored in `eeg-raw-edf`
2. the exam row is created with `status = processing_ai`
3. the worker finds that row
4. it downloads the EDF, runs inference, and exports:
   - `exams/<exam_code>/manifest.json`
   - `exams/<exam_code>/chunks/*.json`
5. it uploads the payload to `eeg-viewer-payloads`
6. it flips the exam to `ready`

Reviews and annotations are not touched by this worker.

## Required files on the worker machine

The machine that runs the worker needs this repo plus the model artifacts already used locally:

- `features.pkl`
- `model_mega.pkl` or `model.pkl`
- any Python dependencies already required by the local pipeline:
  - `mne`
  - `numpy`
  - `scipy`

## Required environment

Create a `.env` file from [.env.example](/Users/rodmag/.codex/worktrees/83e9/eeg-ml/.env.example):

```bash
SUPABASE_URL=...
SUPABASE_PUBLISHABLE_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_VIEWER_PAYLOAD_BUCKET=eeg-viewer-payloads
SUPABASE_RAW_EDF_BUCKET=eeg-raw-edf
```

## Manual commands

Process one exam:

```bash
python3 scripts/process_uploaded_edfs.py --exam-code YOUR_EXAM_CODE
```

Poll the queue continuously:

```bash
bash scripts/process_uploaded_edfs.sh
```

## Suggested production setup

Use a small Linux VM or container that keeps this worker alive continuously.

Recommended defaults:

- interval: `30` seconds
- batch limit: `10`
- chunk samples: `40000`

## systemd example

The repo includes a ready example service:

- [scripts/process_uploaded_edfs.service](/Users/rodmag/.codex/worktrees/83e9/eeg-ml/scripts/process_uploaded_edfs.service)

Typical install flow on the server:

```bash
sudo cp scripts/process_uploaded_edfs.service /etc/systemd/system/neurogram-edf-worker.service
sudo systemctl daemon-reload
sudo systemctl enable neurogram-edf-worker
sudo systemctl start neurogram-edf-worker
sudo systemctl status neurogram-edf-worker
```

## Operational note

This worker is intentionally separate from Netlify Functions.

The current pipeline depends on:

- `mne`
- heavy numerical Python packages
- local model artifacts

That makes a persistent Python worker a safer fit than trying to force full inference into a short-lived Netlify function.
