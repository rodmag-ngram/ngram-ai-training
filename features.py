"""
features.py — Extrai features espectrais, temporais e topográficas.

Por canal:
  - Potência relativa em 5 bandas: delta, theta, alpha, beta, gamma
  - Variância
  - Taxa de zero-crossings

Resumo topográfico adicional:
  - Assimetria hemisférica em pares esquerda/direita por banda
  - Média por região (frontal, temporal, central, parietal, occipital) por banda
  - Correlação média intra-hemisfério e inter-hemisfério
  - Razões globais lentas/rápidas e frontal/posterior
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.signal import welch

ROOT     = Path(__file__).parent
IN_PKL   = ROOT / "dataset.pkl"
OUT_PKL  = ROOT / "features.pkl"

SFREQ    = 250.0
NPERSEG  = 256   # ~1s de janela para Welch

BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 70.0),
}
BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]
CHANNELS = [
    "FP1", "F7", "T3", "T5", "O1", "F3", "C3", "P3", "FZ", "CZ", "PZ",
    "FP2", "F8", "T4", "T6", "O2", "F4", "C4", "P4", "CA1", "CA2", "CA3",
]
CHANNEL_INDEX = {name: idx for idx, name in enumerate(CHANNELS)}
HEMISPHERE_PAIRS = [
    ("FP1", "FP2"),
    ("F3", "F4"),
    ("F7", "F8"),
    ("C3", "C4"),
    ("P3", "P4"),
    ("O1", "O2"),
    ("T3", "T4"),
    ("T5", "T6"),
]
REGIONS = {
    "frontal": ["FP1", "F3", "F7", "FZ", "FP2", "F4", "F8"],
    "temporal": ["T3", "T5", "T4", "T6"],
    "central": ["C3", "CZ", "C4", "CA1", "CA2", "CA3"],
    "parietal": ["P3", "PZ", "P4"],
    "occipital": ["O1", "O2"],
}
LEFT_HEMISPHERE = ["FP1", "F7", "T3", "T5", "O1", "F3", "C3", "P3"]
RIGHT_HEMISPHERE = ["FP2", "F8", "T4", "T6", "O2", "F4", "C4", "P4"]


def _band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    """Potência média na banda [low, high] Hz."""
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return 0.0
    return float(np.mean(psd[mask]))


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _channel_indices(names: list[str]) -> list[int]:
    return [CHANNEL_INDEX[name] for name in names if name in CHANNEL_INDEX]


def extract_window(signal: np.ndarray, sfreq: float = SFREQ) -> np.ndarray:
    """
    signal: (n_channels, n_samples)
    Retorna vetor de features (n_channels * 7,) normalizado por z-score.
    """
    n_ch = signal.shape[0]
    feats = []
    per_channel_band = []

    for ch in range(n_ch):
        x = signal[ch].astype(np.float64)

        # PSD via Welch
        freqs, psd = welch(x, fs=sfreq, nperseg=NPERSEG, window="hann")

        # Potência total para normalização relativa
        total = float(np.sum(psd)) + 1e-12

        # 5 bandas (potência relativa)
        band_feats = [
            _band_power(freqs, psd, lo, hi) / total
            for (lo, hi) in [BANDS[b] for b in BAND_ORDER]
        ]
        per_channel_band.append(band_feats)

        # Variância
        variance = float(np.var(x))

        # Zero-crossing rate (cruza por amostra)
        zcr = float(np.sum(np.diff(np.sign(x)) != 0)) / len(x)

        feats.extend(band_feats + [variance, zcr])

    band_matrix = np.array(per_channel_band, dtype=np.float32)

    topo_feats = []

    for left_name, right_name in HEMISPHERE_PAIRS:
        left_idx = CHANNEL_INDEX[left_name]
        right_idx = CHANNEL_INDEX[right_name]
        diffs = band_matrix[left_idx] - band_matrix[right_idx]
        topo_feats.extend(diffs.tolist())

    for region_channels in REGIONS.values():
        idxs = _channel_indices(region_channels)
        region_mean = band_matrix[idxs].mean(axis=0) if idxs else np.zeros(len(BAND_ORDER), dtype=np.float32)
        topo_feats.extend(region_mean.tolist())

    corr = np.corrcoef(signal)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    left_idxs = _channel_indices(LEFT_HEMISPHERE)
    right_idxs = _channel_indices(RIGHT_HEMISPHERE)
    inter_pairs = [(li, ri) for li, ri in zip(left_idxs, right_idxs)]
    left_corr = [corr[i, j] for i in left_idxs for j in left_idxs if i < j]
    right_corr = [corr[i, j] for i in right_idxs for j in right_idxs if i < j]
    inter_corr = [corr[i, j] for i, j in inter_pairs]
    topo_feats.extend([
        _safe_mean(left_corr),
        _safe_mean(right_corr),
        _safe_mean(inter_corr),
    ])

    delta_power = float(np.mean(band_matrix[:, 0]))
    theta_power = float(np.mean(band_matrix[:, 1]))
    alpha_power = float(np.mean(band_matrix[:, 2]))
    beta_power = float(np.mean(band_matrix[:, 3]))
    gamma_power = float(np.mean(band_matrix[:, 4]))
    slow_power = delta_power + theta_power
    fast_power = alpha_power + beta_power + gamma_power + 1e-8

    frontal_idxs = _channel_indices(REGIONS["frontal"])
    posterior_idxs = _channel_indices(REGIONS["parietal"] + REGIONS["occipital"])
    frontal_slow = float(np.mean(band_matrix[frontal_idxs, 0] + band_matrix[frontal_idxs, 1])) if frontal_idxs else 0.0
    posterior_alpha = float(np.mean(band_matrix[posterior_idxs, 2])) if posterior_idxs else 0.0

    topo_feats.extend([
        slow_power / fast_power,
        delta_power / (alpha_power + 1e-8),
        frontal_slow / (posterior_alpha + 1e-8),
    ])

    return np.array(feats + topo_feats, dtype=np.float32)


def extract(data=None):
    """
    Extrai features de todos os windows em `data` (ou carrega dataset.pkl).
    Retorna (X, y, meta) onde:
      X    : (n_windows, n_features) float32, z-score normalizado
      y    : (n_windows,) str array de labels
      meta : lista de dicts com exam_id, patient_id, window_idx, etc.
    """
    if data is None:
        print(f"Carregando {IN_PKL} ...")
        with open(IN_PKL, "rb") as f:
            data = pickle.load(f)

    print(f"Extraindo features de {len(data)} janelas ...")

    X_list = []
    y_list = []
    meta   = []

    for i, w in enumerate(data):
        if i % 1000 == 0:
            print(f"  {i}/{len(data)}", flush=True)

        feats = extract_window(w["signal"], sfreq=w["sfreq"])
        X_list.append(feats)
        y_list.append(w["label"])
        meta.append({
            "exam_id":          w["exam_id"],
            "patient_id":       w["patient_id"],
            "window_idx":       w["window_idx"],
            "window_start":     w["window_start"],
            "window_end":       w["window_end"],
            "label":            w["label"],
            "labels_per_rater": w["labels_per_rater"],
        })

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)

    # Z-score global por feature (fit nos dados completos)
    mu  = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X   = (X - mu) / std

    print(f"X shape: {X.shape}  |  features por janela: {X.shape[1]}")
    print(f"Labels únicas: {np.unique(y).tolist()}")

    return X, y, meta, mu, std


if __name__ == "__main__":
    X, y, meta, mu, std = extract()

    print("\nEstatísticas por label:")
    for lbl in sorted(set(y)):
        mask = y == lbl
        print(f"  {lbl:10s}: {mask.sum():5d} janelas")

    print(f"\nSalvando {OUT_PKL} ...")
    with open(OUT_PKL, "wb") as f:
        pickle.dump({"X": X, "y": y, "meta": meta, "mu": mu, "std": std}, f, protocol=4)
    print("Pronto.")
