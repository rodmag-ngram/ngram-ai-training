"""
visualizer.py — Gera HTML interativo comparando anotações AI × médicas.

Baseado no viewer/phenomenon_viewer.html.
Adiciona um painel de comparação com 4 faixas coloridas:
  AI | Elaine | Amanda | Marina

Uso:
  python3 visualizer.py                          # exam mais variado do conjunto de teste
  python3 visualizer.py PAT-U2XE-DTG5-VLMC-2_1  # exam específico
"""

import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT      = Path(__file__).parent
FEAT_PKL  = ROOT / "features.pkl"
MODEL_PKL = ROOT / "model.pkl"
DATASET   = ROOT / "dataset.pkl"
TEST_EXAMS = ROOT / "test_exams.txt"
TEMPLATE  = ROOT / "viewer" / "phenomenon_viewer.html"
OUT_DIR   = ROOT / "viewer"

TARGET_SFREQ = 50.0  # Hz do viewer
WINDOW_S     = 10.0
RATERS       = ["elaine", "amanda", "marina"]

LABEL_COLORS = {
    "normal":  "#4CAF50",
    "gpd":     "#2196F3",
    "lpd":     "#9C27B0",
    "lrda":    "#FF9800",
    "grda":    "#FF5722",
    "seizure": "#F44336",
    "other":   "#9E9E9E",
}


def pick_exam(test_exam_ids, meta):
    """Escolhe o exame de teste com mais classes únicas de label."""
    exam_labels = defaultdict(set)
    for m in meta:
        if m["exam_id"] in test_exam_ids:
            exam_labels[m["exam_id"]].add(m["label"])
    # Prefere exames com labels variados e pelo menos seizure ou gpd
    def score(eid):
        lbls = exam_labels[eid]
        interesting = len({"gpd","lpd","seizure","lrda","grda"} & lbls)
        return (interesting, len(lbls))
    return max(exam_labels, key=score)


def reconstruct_signal(exam_id, dataset):
    """Reconstrói sinal completo concatenando janelas na ordem."""
    windows = [w for w in dataset if w["exam_id"] == exam_id]
    windows.sort(key=lambda w: w["window_idx"])
    signal_parts = [w["signal"] for w in windows]  # each: (n_ch, n_samples)
    signal = np.concatenate(signal_parts, axis=1)  # (n_ch, total_samples)
    sfreq  = windows[0]["sfreq"]
    ch_names = windows[0]["ch_names"]
    return signal, sfreq, ch_names, windows


def downsample(signal, orig_sfreq, target_sfreq):
    """Downsampling simples por decimação."""
    factor = int(orig_sfreq / target_sfreq)
    return signal[:, ::factor]


def get_ai_predictions(exam_id, meta, X, model, le):
    """Retorna lista de (window_idx, label_pred) para o exame."""
    idxs = [i for i, m in enumerate(meta) if m["exam_id"] == exam_id]
    if not idxs:
        return []
    X_exam = X[idxs]
    enc = model.predict(X_exam)
    preds = le.inverse_transform(enc)
    return [(meta[i]["window_idx"], preds[j]) for j, i in enumerate(idxs)]


def windows_to_segments(windows_meta, label_key):
    """
    Converte janelas em segmentos contíguos do mesmo label.
    label_key: 'label' (consenso) ou rater name
    """
    segments = []
    cur_label = None
    cur_start = None

    for m in sorted(windows_meta, key=lambda x: x["window_idx"]):
        if label_key in RATERS:
            lbl = m["labels_per_rater"].get(label_key, "normal")
        else:
            lbl = m[label_key]

        t_start = m["window_start"]
        t_end   = m["window_end"]

        if lbl != cur_label:
            if cur_label is not None:
                segments.append((cur_start, t_start, cur_label))
            cur_label = lbl
            cur_start = t_start

    if cur_label is not None:
        segments.append((cur_start, windows_meta[-1]["window_end"], cur_label))

    return segments


def ai_preds_to_segments(pred_pairs, meta_exam):
    """Converte predições AI em segmentos."""
    idx_map = {m["window_idx"]: m for m in meta_exam}
    segments = []
    cur_label = None
    cur_start = None

    for widx, lbl in sorted(pred_pairs):
        m = idx_map.get(widx)
        if m is None:
            continue
        t_start, t_end = m["window_start"], m["window_end"]
        if lbl != cur_label:
            if cur_label is not None:
                segments.append((cur_start, t_start, cur_label))
            cur_label = lbl
            cur_start = t_start

    if cur_label is not None:
        last_m = idx_map[max(idx_map)]
        segments.append((cur_start, last_m["window_end"], cur_label))

    return segments


def build_comparison_js(tracks: dict, duration: float) -> str:
    """
    tracks: {'AI': [(t0,t1,label), ...], 'Elaine': [...], ...}
    Retorna bloco JS que injeta o painel de comparação abaixo do EEG.
    """
    tracks_json = json.dumps({
        name: [{"t0": t0, "t1": t1, "label": lbl} for t0, t1, lbl in segs]
        for name, segs in tracks.items()
    })
    colors_json = json.dumps(LABEL_COLORS)
    track_names = list(tracks.keys())

    return f"""
// ── Comparison Panel ──────────────────────────────────────────────────
(function() {{
  const TRACKS = {tracks_json};
  const COLORS = {colors_json};
  const TRACK_NAMES = {json.dumps(track_names)};
  const DUR = {duration:.3f};
  const ROW_H = 28;
  const LABEL_W = 72;

  // Create panel container
  const panel = document.createElement('div');
  panel.id = 'cmp-panel';
  panel.style.cssText = 'background:#1a1a2e;border-top:2px solid #444;flex-shrink:0;overflow:hidden;';
  panel.style.height = (ROW_H * TRACK_NAMES.length + 22) + 'px';

  // Header
  const hdr = document.createElement('div');
  hdr.style.cssText = 'display:flex;align-items:center;padding:3px 10px 2px;background:#111;';
  hdr.innerHTML = '<span style="font-size:10px;color:#aaa;font-weight:600;letter-spacing:.08em;">COMPARAÇÃO: ANOTAÇÕES</span>';

  // Legend
  const legend = document.createElement('div');
  legend.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;margin-left:auto;';
  Object.entries(COLORS).forEach(([lbl, col]) => {{
    legend.innerHTML += `<span style="font-size:9px;color:${{col}};white-space:nowrap;">■ ${{lbl}}</span>`;
  }});
  hdr.appendChild(legend);
  panel.appendChild(hdr);

  // Canvas
  const canvas = document.createElement('canvas');
  canvas.id = 'cmp-canvas';
  canvas.style.cssText = 'display:block;width:100%;';
  panel.appendChild(canvas);

  // Insert after timeline-wrap
  const tlWrap = document.getElementById('timeline-wrap');
  tlWrap.parentNode.insertBefore(panel, tlWrap.nextSibling);

  function drawComparison() {{
    const W = panel.clientWidth;
    const H = ROW_H * TRACK_NAMES.length;
    canvas.width  = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    const plotW = W - LABEL_W;

    TRACK_NAMES.forEach((name, ri) => {{
      const y0 = ri * ROW_H;
      const segs = TRACKS[name] || [];

      // Row background
      ctx.fillStyle = ri % 2 === 0 ? '#16213e' : '#0f3460';
      ctx.fillRect(0, y0, W, ROW_H);

      // Row label
      ctx.fillStyle = '#ccc';
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(name, 6, y0 + ROW_H/2 + 4);

      // Segments
      segs.forEach(seg => {{
        const x0 = LABEL_W + (seg.t0 / DUR) * plotW;
        const x1 = LABEL_W + (seg.t1 / DUR) * plotW;
        const w  = Math.max(1, x1 - x0);
        const color = COLORS[seg.label] || '#888';
        ctx.fillStyle = color + 'cc';
        ctx.fillRect(x0, y0 + 3, w, ROW_H - 6);
        // Label text if wide enough
        if (w > 28) {{
          ctx.fillStyle = '#fff';
          ctx.font = '9px sans-serif';
          ctx.textAlign = 'left';
          ctx.fillText(seg.label, x0 + 2, y0 + ROW_H/2 + 3);
        }}
      }});

      // Row border
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(0, y0, W, ROW_H);

      // Vertical cursor line (synced with viewer startTime)
      const cx = LABEL_W + (startTime / DUR) * plotW;
      ctx.strokeStyle = 'rgba(74,144,217,0.8)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3,2]);
      ctx.beginPath(); ctx.moveTo(cx, y0); ctx.lineTo(cx, y0+ROW_H); ctx.stroke();
      ctx.setLineDash([]);
    }});

    // Time axis
    ctx.fillStyle = '#667';
    ctx.font = '8px sans-serif';
    ctx.textAlign = 'center';
    for (let t = 0; t <= DUR; t += 60) {{
      const x = LABEL_W + (t / DUR) * plotW;
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }}
  }}

  // Hook into existing navigate to keep cursor in sync
  const origNavigate = window.navigate || null;
  function patchNavigate() {{
    if (typeof navigate === 'function') {{
      const _nav = navigate;
      window._navigate = _nav;
      window.navigate = function(t) {{ _nav(t); drawComparison(); }};
    }}
  }}

  // Click on comparison panel to navigate
  canvas.addEventListener('click', (e) => {{
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const plotW = canvas.width - LABEL_W;
    if (x < LABEL_W) return;
    const t = ((x - LABEL_W) / plotW) * DUR;
    navigate(Math.max(0, t - windowSize/2));
    drawComparison();
  }});

  window.addEventListener('resize', drawComparison);

  // Patch draw to also refresh comparison
  const origDraw = window.draw;
  window.draw = function() {{ origDraw(); drawComparison(); }};

  setTimeout(drawComparison, 200);
}})();
"""


def generate(exam_id=None):
    print("Carregando artefatos ...")
    with open(FEAT_PKL, "rb") as f:
        bundle = pickle.load(f)
    with open(MODEL_PKL, "rb") as f:
        mb = pickle.load(f)
    with open(DATASET, "rb") as f:
        dataset = pickle.load(f)

    X, meta = bundle["X"], bundle["meta"]
    model, le = mb["model"], mb["label_encoder"]
    test_exam_ids = set(TEST_EXAMS.read_text().splitlines())

    if exam_id is None:
        exam_id = pick_exam(test_exam_ids, meta)
    print(f"Exame selecionado: {exam_id}")

    # ── Sinal ─────────────────────────────────────────────────────────────────
    signal, sfreq, ch_names, windows = reconstruct_signal(exam_id, dataset)
    signal_ds = downsample(signal, sfreq, TARGET_SFREQ)
    duration = signal.shape[1] / sfreq

    print(f"Duração: {duration:.1f}s  |  {signal.shape[0]} canais  |  "
          f"{signal_ds.shape[1]} amostras @ {TARGET_SFREQ}Hz")

    # ── Predições AI ──────────────────────────────────────────────────────────
    ai_preds = get_ai_predictions(exam_id, meta, X, model, le)
    meta_exam = [m for m in meta if m["exam_id"] == exam_id]
    meta_exam.sort(key=lambda m: m["window_idx"])

    # ── Segmentos por fonte ───────────────────────────────────────────────────
    tracks = {
        "AI":     ai_preds_to_segments(ai_preds, meta_exam),
        "Elaine": windows_to_segments(meta_exam, "elaine"),
        "Amanda": windows_to_segments(meta_exam, "amanda"),
        "Marina": windows_to_segments(meta_exam, "marina"),
    }

    print("Segmentos por fonte:")
    for name, segs in tracks.items():
        unique_lbls = list({s[2] for s in segs})
        print(f"  {name:8s}: {len(segs)} segmentos, labels={unique_lbls}")

    # ── RAW inline (arredonda para 2 casas) ───────────────────────────────────
    raw_rows = []
    for ch in range(signal_ds.shape[0]):
        vals = signal_ds[ch].tolist()
        vals_str = "[" + ",".join(f"{v:.2f}" for v in vals) + "]"
        raw_rows.append(vals_str)
    raw_js = "[\n" + ",\n".join(raw_rows) + "\n]"

    n_samples = signal_ds.shape[1]
    ch_names_js = json.dumps(ch_names)

    # ── Annotations legadas (usadas pelo viewer original) ─────────────────────
    # Pré-popula com segmentos da AI como annotations para mostrar na timeline
    legacy_anns = []
    ann_id = 1
    for t0, t1, lbl in tracks["AI"]:
        if lbl == "normal":
            continue  # não polui a timeline com normal
        legacy_anns.append({
            "id": ann_id, "t0": t0, "t1": t1,
            "type": f"AI:{lbl}", "deleted": False
        })
        ann_id += 1
    legacy_anns_js = json.dumps(legacy_anns)

    # ── Lê template e substitui placeholders ─────────────────────────────────
    template = TEMPLATE.read_text(encoding="utf-8")

    # Substituições no bloco <script>
    template = re.sub(
        r'const CHANNELS\s*=\s*\[.*?\];',
        f'const CHANNELS = {ch_names_js};',
        template
    )
    template = re.sub(
        r'const SFREQ\s*=\s*[\d.]+;',
        f'const SFREQ = {TARGET_SFREQ};',
        template
    )
    template = re.sub(
        r'const DURATION\s*=\s*[\d.]+;',
        f'const DURATION = {duration:.3f};',
        template
    )
    template = re.sub(
        r'const RAW\s*=\s*\[[\s\S]*?\];(?=\s*\n)',
        f'const RAW = {raw_js};',
        template
    )
    template = re.sub(
        r'let annotations\s*=\s*\[\];',
        f'let annotations = {legacy_anns_js};\nlet annIdCounter = {ann_id};',
        template
    )
    # Título
    template = template.replace(
        "Phenomenon Viewer — PAT-ZF5A-BYWS-PKMO",
        f"Phenomenon Viewer — {exam_id}"
    )
    # Export patient name
    template = re.sub(
        r"patient:'PAT-ZF5A-BYWS-PKMO'",
        f"patient:'{exam_id}'",
        template
    )
    template = re.sub(
        r"a\.download='review_PAT-ZF5A-BYWS-PKMO\.json'",
        f"a.download='review_{exam_id}.json'",
        template
    )

    # Injeta painel de comparação antes de </script></body>
    comparison_js = build_comparison_js(tracks, duration)
    template = template.replace(
        "// ── Init ─────────────────────────────────────────────────────────────",
        comparison_js + "\n// ── Init ─────────────────────────────────────────────────────────────"
    )

    # Salva
    out_path = OUT_DIR / f"comparison_{exam_id}.html"
    out_path.write_text(template, encoding="utf-8")
    print(f"\nHTML gerado: {out_path}")
    print(f"Tamanho: {out_path.stat().st_size / 1024:.0f} KB")
    return out_path


if __name__ == "__main__":
    exam_id = sys.argv[1] if len(sys.argv) > 1 else None
    generate(exam_id)
