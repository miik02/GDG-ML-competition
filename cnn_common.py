import numpy as np
import torch
import torch.nn as nn

CLIP = 8.0
GROUPS = ("A", "B", "C")
IN_CH = 12  # 4 señal + 4 máscara + 4 log-escala + 4 meta  # 4 señal + 4 máscara NaN + 4 meta (sample_rate, grupo A/B/C)


def _mid(a):
    """Mediana (inferior) ignorando NaN sobre el último eje, vía sort: rápida y válida para lotes."""
    s = np.sort(a, axis=-1)  # los NaN quedan al final
    cnt = (~np.isnan(a)).sum(-1)
    idx = np.maximum(cnt - 1, 0) // 2
    return np.take_along_axis(s, idx[..., None], -1)


def normalize(x):
    """(..., 4, L) con NaN -> (..., 8, L): señal robusta + máscara de NaN."""
    nan = np.isnan(x)
    med = _mid(x)
    mad = _mid(np.abs(x - med)) * 1.4826
    z = (x - med) / (mad + 1e-3)       # quita offset (y el atajo de `level`) y la escala
    z = CLIP * np.tanh(z / CLIP)       # los picos saturan en vez de aplastar la señal
    z = np.where(nan, 0.0, z).astype(np.float32)
    return np.concatenate([z, nan.astype(np.float32)], axis=-2)


def meta_vector(meta):
    sr = 1.0 if float(meta.get("sample_rate", 1)) >= 1.5 else 0.0
    g = meta.get("group", "A")
    return np.array([sr] + [float(g == k) for k in GROUPS], np.float32)


def build_input(series, metav):
    """series (B,4,L) o (4,L); metav (B,4) o (4,)."""
    z = normalize(series)
    m = np.broadcast_to(metav[..., :, None], metav.shape + (z.shape[-1],))
    return np.concatenate([z, m], axis=-2).astype(np.float32)


def _gap_len(rng):
    return int(np.clip(round(rng.lognormal(0.2, 1.6)), 1, 24))  # aprox. p50=1, p90≈10, máx 24


def augment(x, rng, p_nan=0.9, p_spike=0.4):
    """Simula el test: ruido en casi todas las muestras (~0.3 MAD), más NaN y picos. x: (B,4,L) crudo."""
    x = x.copy()
    B, C, L = x.shape
    for b in range(B):
        # 1) ruido blanco proporcional al MAD, nivel calibrado con diag4 (~0.3)
        if rng.random() < 0.95:
            med = np.nanmedian(x[b], axis=1, keepdims=True)
            mad = np.nanmedian(np.abs(x[b] - med)) * 1.4826
            x[b] += rng.normal(0, 1, x[b].shape).astype(np.float32) * (rng.uniform(0.1, 0.6) * mad)
        # 2) picos
        if rng.random() < p_spike:
            for _ in range(int(rng.integers(1, 20))):
                n = 1 if rng.random() < 0.7 else int(rng.integers(2, 5))
                s, c = int(rng.integers(0, L)), int(rng.integers(C))
                amp = 10 ** (1 + 2 * rng.random() ** 3) * rng.choice([-1.0, 1.0])
                x[b, c, s:s + n] += amp
        # 3) huecos de NaN
        if rng.random() < p_nan:
            for _ in range(int(rng.integers(1, 12))):
                n = _gap_len(rng)
                s, c = int(rng.integers(0, L - n + 1)), int(rng.integers(C))
                x[b, c, s:s + n] = np.nan
    return x


class Net(nn.Module):
    def __init__(self, w=32, in_ch=IN_CH, n_classes=8, drop=0.2):
        super().__init__()

        def blk(i, o, k, s):
            return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2, bias=False), nn.BatchNorm1d(o), nn.ReLU(inplace=True))
        self.f = nn.Sequential(blk(in_ch, w, 7, 1), blk(w, w * 3 // 2, 5, 2), blk(w * 3 // 2, w * 2, 5, 2),
                               blk(w * 2, w * 3, 3, 2), blk(w * 3, w * 4, 3, 2))
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(w * 8, n_classes))

    def forward(self, x):
        h = self.f(x)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))