# Oracle VM Setup For EDF Worker

This is the simplest path to keep the uploaded-EDF worker running continuously on Oracle Cloud Always Free.

## Recommended shape

Use an **Always Free** compute instance in your **home region**.

Recommended for this worker:

- `VM.Standard.A1.Flex`
- start with `2 OCPU / 12 GB RAM`

This is preferred over the tiny micro instance because the EEG worker uses scientific Python packages and model inference.

Oracle documents that Always Free compute must be created in the tenancy home region, and that A1 capacity can be temporarily unavailable in some availability domains.

## 1. Create the VM

In OCI Console:

1. Go to `Compute`
2. Click `Create instance`
3. Choose:
   - image: `Ubuntu 22.04`
   - shape: `VM.Standard.A1.Flex`
   - shape config: `2 OCPU / 12 GB`
4. Add your SSH public key
5. Create the instance

## 2. Connect over SSH

Oracle's SSH pattern is:

```bash
ssh -i /path/to/private_key opc@YOUR_PUBLIC_IP
```

## 3. Install system packages

On the VM:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip build-essential libopenblas-dev libomp-dev
```

## 4. Clone the repo

```bash
git clone https://github.com/rodmag-ngram/ngram-ai-training.git
cd ngram-ai-training
```

## 5. Create Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-worker.txt
```

## 6. Create `.env`

Copy the example:

```bash
cp .env.example .env
```

Then fill:

```bash
SUPABASE_URL=...
SUPABASE_PUBLISHABLE_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_VIEWER_PAYLOAD_BUCKET=eeg-viewer-payloads
SUPABASE_RAW_EDF_BUCKET=eeg-raw-edf
```

## 7. Put the model artifacts on the VM

The worker needs the same local artifacts already used by the repo:

- `features.pkl`
- `model_mega.pkl` or `model.pkl`

Place them in the project root beside the Python files.

## 8. Test one exam manually

```bash
source .venv/bin/activate
python3 scripts/process_uploaded_edfs.py --exam-code YOUR_EXAM_CODE
```

## 9. Start the continuous worker

```bash
source .venv/bin/activate
bash scripts/process_uploaded_edfs.sh
```

## 10. Run it as a service

Edit the service file paths if your clone lives somewhere else, then:

```bash
sudo cp scripts/process_uploaded_edfs.service /etc/systemd/system/neurogram-edf-worker.service
sudo systemctl daemon-reload
sudo systemctl enable neurogram-edf-worker
sudo systemctl start neurogram-edf-worker
sudo systemctl status neurogram-edf-worker
```

## Notes

- If OCI says there's no A1 capacity, try another availability domain or try again later.
- The tiny Always Free micro shape is usually not the best choice for this scientific Python worker.
