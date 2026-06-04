"""
ingest.py — Segmenta EDFs em janelas de 10s e atribui labels por votação majoritária.

Saída: dataset.pkl com lista de dicts:
  {
    'exam_id': str,         # stem do EDF (ex: PAT-ZF5A-BYWS-PKMO_3)
    'patient_id': str,      # ID do paciente (ex: PAT-ZF5A-BYWS-PKMO)
    'window_idx': int,
    'window_start': float,  # segundos no EDF
    'window_end': float,
    'label': str,           # label consenso
    'labels_per_rater': dict,  # {'elaine': ..., 'amanda': ..., 'marina': ...}
    'signal': np.ndarray,   # shape (n_channels, n_samples)
    'ch_names': list[str],
    'sfreq': float,
  }
"""

import csv
import json
import pickle
import re
import warnings
from collections import defaultdict
from pathlib import Path

import mne
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA       = ROOT / "data"
EXAMS_DIR  = DATA / "exams"
ELAINE_DIR = DATA / "elaine"
AMANDA_DIR = DATA / "amanda"
MARINA_DIR = DATA / "marina"
MAPPING    = ROOT / "exam_mapping.csv"
OUT_PKL    = ROOT / "dataset.pkl"

WINDOW_S   = 10.0  # segundos por janela
LABELS_6   = {"gpd", "lpd", "lrda", "grda", "seizure", "normal"}
PRIORITY   = {"seizure": 5, "gpd": 4, "lpd": 4, "lrda": 3, "grda": 3,
              "other": 2, "normal": 1}


# ── Normalização de label ─────────────────────────────────────────────────────
def normalize_label(raw: str) -> str:
    """Mapeia qualquer variação de label para uma das 6 classes canônicas."""
    s = raw.strip().lower()
    # Remove prefixos START/END/Start/End
    s = re.sub(r'^(start|end|star)\s*:\s*', '', s)
    s = s.strip()
    # Primeiro token (antes de vírgula, espaço ou parêntese)
    token = re.split(r'[\s,\(]', s)[0]
    if token.startswith('gpd') or token == 'gdra':  # GDRA é typo de GRDA? não — é typo de GPD? ambíguo; tratamos como gpd
        # na verdade GDRA aparece em Amanda como "GDRA" (typo de GRDA)
        pass
    MAP = {
        'gpd':     'gpd',
        'lpd':     'lpd',
        'lrda':    'lrda',
        'grda':    'grda',
        'gdra':    'grda',   # typo de GRDA em Amanda
        'seizure': 'seizure',
        'normal':  'normal',
        'bird':    'other',
        'bipds':   'other',
        'birds':   'other',
        'tw':      'other',
        'others':  'other',
        'other':   'other',
        'bs':      'other',
        'burst':   'other',
        'suppression': 'other',
        'artifact':    'other',
        'artefato':    'other',
        'artifacts':   'other',
        'artefacts':   'other',
        'eletrodo':    'other',
        'electrode':   'other',
        'muscular':    'other',
        'sweat':       'other',
        'movment':     'other',
        'movement':    'other',
        'falha':       'other',
        'epileptiform':'other',
        'frda':        'other',
        'status':      'other',
        'discharge':   'other',
        'dischages':   'other',
        'discharges':  'other',
        'epileptic':   'other',
        'bird,':       'other',
    }
    result = MAP.get(token)
    if result is None:
        # fallback: procura substring
        for key, val in MAP.items():
            if key in s:
                return val
        return 'other'
    return result


# ── Parsers de anotação → lista de (onset_s, end_s, label) ───────────────────

def parse_elaine(path: Path) -> list[tuple[float, float, str]]:
    """Elaine: {onset, duration, text} — já normalizadas."""
    d = json.loads(path.read_text())
    segments = []
    for a in d.get("annotations", []):
        onset    = float(a["onset"])
        duration = float(a["duration"])
        label    = normalize_label(a["text"])
        segments.append((onset, onset + duration, label))
    return segments


def parse_start_end_json(path: Path) -> list[tuple[float, float, str]]:
    """
    Amanda/Marina: aceita dois formatos:
      - pares START/END explícitos
      - eventos repetidos que funcionam como toggle implícito (início/fim)
    """
    events = json.loads(path.read_text())
    segments = []
    stack: dict[str, float] = {}
    toggle_open: dict[str, float] = {}

    for ev in events:
        raw = str(ev["label"])
        t = float(ev["time"])
        lower = raw.strip().lower()

        is_start = bool(re.match(r'^(start|star)\s*:', lower))
        is_end = bool(re.match(r'^end\b', lower))

        canon = normalize_label(raw)

        if is_start:
            if canon in stack:
                segments.append((stack[canon], t, canon))
            stack[canon] = t
            continue

        if is_end:
            if canon in stack:
                segments.append((stack.pop(canon), t, canon))
            elif canon in toggle_open:
                segments.append((toggle_open.pop(canon), t, canon))
            continue

        if canon in toggle_open:
            start_t = toggle_open.pop(canon)
            if t > start_t:
                segments.append((start_t, t, canon))
        else:
            toggle_open[canon] = t

    for canon, start_t in stack.items():
        if canon in toggle_open and toggle_open[canon] > start_t:
            segments.append((start_t, toggle_open.pop(canon), canon))

    return segments


def resolve_edf_path(edf_file: str) -> tuple[str, Path]:
    """Corrige typos simples do mapping sem alterar o CSV."""
    candidates = [edf_file]
    if ".ed.edf" in edf_file:
        candidates.append(edf_file.replace(".ed.edf", ".edf"))

    for candidate in candidates:
        path = EXAMS_DIR / candidate
        if path.exists():
            return path.stem, path

    return Path(edf_file).stem, EXAMS_DIR / edf_file


def resolve_elaine_path(exam_id: str, row: dict) -> Path:
    """Elaine real está por exam_id, mas tenta fallback pelo mapping."""
    candidates = [
        ELAINE_DIR / f"{exam_id}.json",
        ELAINE_DIR / row.get("elaine_json", ""),
    ]
    for candidate in candidates:
        if candidate.name and candidate.exists():
            return candidate
    return candidates[0]


def resolve_rater_path(base_dir: Path, filename: str) -> Path:
    """Retorna caminho ou placeholder quando o mapping está claramente quebrado."""
    filename = (filename or "").strip()
    if not filename:
        return base_dir / "__missing__.json"
    if "missing" in filename.lower():
        return base_dir / filename
    return base_dir / filename


def load_annotations(exam_id: str, row: dict) -> tuple[dict[str, list], dict[str, str]]:
    """Retorna {'elaine': [...], 'amanda': [...], 'marina': [...]}."""
    elaine_path = resolve_elaine_path(exam_id, row)
    amanda_path = resolve_rater_path(AMANDA_DIR, row["amanda_json"])
    marina_path = resolve_rater_path(MARINA_DIR, row["marina_json"])

    result = {}
    issues = {}
    for name, path, parser in [
        ("elaine", elaine_path, parse_elaine),
        ("amanda", amanda_path, parse_start_end_json),
        ("marina", marina_path, parse_start_end_json),
    ]:
        if path.exists():
            try:
                result[name] = parser(path)
            except Exception as e:
                print(f"  [WARN] {name} parse error for {exam_id}: {e}")
                result[name] = []
                issues[name] = "parse_error"
        else:
            print(f"  [WARN] missing {name} file: {path.name}")
            result[name] = []
            issues[name] = "missing"
    return result, issues


# ── Votação por janela ────────────────────────────────────────────────────────

def label_for_window(
    win_start: float,
    win_end: float,
    annotations: dict[str, list],
) -> tuple[str, dict[str, str]]:
    """
    Para cada médica: label que cobre mais tempo na janela.
    Consenso: label mais votado entre as 3 (maioria simples).
    Empate → label mais patológica (seizure > gpd/lpd > lrda/grda > other > normal).
    Janela sem cobertura por nenhuma médica → "normal".
    """
    per_rater: dict[str, str] = {}

    for rater, segs in annotations.items():
        coverage: dict[str, float] = defaultdict(float)
        for (s, e, lbl) in segs:
            overlap = max(0.0, min(e, win_end) - max(s, win_start))
            if overlap > 0:
                coverage[lbl] += overlap
        if coverage:
            per_rater[rater] = max(coverage, key=lambda k: (coverage[k], PRIORITY.get(k, 0)))
        else:
            per_rater[rater] = "normal"

    # Votação
    votes: dict[str, int] = defaultdict(int)
    for lbl in per_rater.values():
        votes[lbl] += 1

    max_votes = max(votes.values())
    candidates = [l for l, v in votes.items() if v == max_votes]
    consensus = max(candidates, key=lambda k: PRIORITY.get(k, 0))

    return consensus, per_rater


# ── Main ──────────────────────────────────────────────────────────────────────

def ingest():
    with open(MAPPING, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Exames no mapeamento: {len(rows)}")

    dataset = []
    label_counts: dict[str, int] = defaultdict(int)
    skipped = 0
    missing_counts: dict[str, int] = defaultdict(int)

    for i, row in enumerate(rows):
        exam_id, edf_path = resolve_edf_path(row["edf_file"])
        patient_id = "_".join(exam_id.split("_")[:-1])  # remove sufixo _N

        if not edf_path.exists():
            print(f"  [SKIP] EDF não encontrado: {row['edf_file']}")
            skipped += 1
            continue

        print(f"[{i+1:3d}/{len(rows)}] {exam_id} ...", end=" ", flush=True)

        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        except Exception as e:
            print(f"ERRO ao carregar EDF: {e}")
            skipped += 1
            continue

        sfreq      = raw.info["sfreq"]
        duration_s = raw.times[-1]
        ch_names   = raw.ch_names
        annotations, issues = load_annotations(exam_id, row)
        for issue_key in issues:
            missing_counts[issue_key] += 1

        n_windows = int(duration_s // WINDOW_S)
        win_labels = []

        for w in range(n_windows):
            win_start = w * WINDOW_S
            win_end   = win_start + WINDOW_S

            # Extrai sinal bruto
            s_idx = int(win_start * sfreq)
            e_idx = int(win_end   * sfreq)
            signal = raw.get_data(start=s_idx, stop=e_idx)  # (n_ch, n_samples)

            # Label por votação
            consensus, per_rater = label_for_window(win_start, win_end, annotations)

            dataset.append({
                "exam_id":          exam_id,
                "patient_id":       patient_id,
                "window_idx":       w,
                "window_start":     win_start,
                "window_end":       win_end,
                "label":            consensus,
                "labels_per_rater": per_rater,
                "signal":           signal.astype(np.float32),
                "ch_names":         ch_names,
                "sfreq":            sfreq,
            })
            label_counts[consensus] += 1
            win_labels.append(consensus)

        print(f"{n_windows} janelas | labels: { {l: win_labels.count(l) for l in set(win_labels)} }")

    print(f"\nTotal janelas: {len(dataset)}")
    print(f"Skipped: {skipped} exames")
    print(f"\nDistribuição de labels:")
    total = sum(label_counts.values())
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {lbl:10s}: {cnt:5d}  ({cnt/total*100:.1f}%)")

    if missing_counts:
        print("\nArquivos de anotação com problema:")
        for name, cnt in sorted(missing_counts.items()):
            print(f"  {name:10s}: {cnt}")

    print(f"\nSalvando {OUT_PKL} ...")
    with open(OUT_PKL, "wb") as f:
        pickle.dump(dataset, f, protocol=4)
    print("Pronto.")


if __name__ == "__main__":
    ingest()
