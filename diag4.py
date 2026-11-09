import numpy as np
from competition_api import load_dataset

tr = load_dataset("data/train_100k.npz"); te = load_dataset("data/test_demo.npz")
Xtr, ytr = np.asarray(tr.series), np.asarray(tr.labels)
Xd, yd = np.asarray(te.series), np.asarray(te.labels)
idx = np.random.default_rng(0).permutation(len(ytr))[:20000]   # aleatorio: el train está ordenado por clase
Xtr, ytr = Xtr[idx], ytr[idx]

def dev(X):
    return np.nanmax(np.abs(X - np.nanmedian(X, 2, keepdims=True)), axis=(1, 2))

Xtr_c, Xd_c = Xtr[dev(Xtr) < 15], Xd[dev(Xd) < 15]       # sin picos grandes
print("muestras limpias: train", len(Xtr_c), " demo", len(Xd_c))

def stats(X):
    med = np.nanmedian(X, 2, keepdims=True)
    mad = np.nanmedian(np.abs(X - med), 2, keepdims=True) * 1.4826
    z = np.clip((X - med) / (mad + 1e-3), -8, 8)
    ac1 = np.nanmean(z[:, :, 1:] * z[:, :, :-1]) / np.nanmean(z ** 2)
    p = (np.abs(np.fft.rfft(np.nan_to_num(z), axis=2)) ** 2).mean((0, 1))[1:]
    return ac1, np.log10(p[:32].sum()), np.log10(p[-64:].sum())

rng = np.random.default_rng(1)
def add_noise(X, s):
    X = X.copy()
    med = np.nanmedian(X, 2, keepdims=True)
    mad = np.nanmedian(np.abs(X - med), axis=(1, 2)) * 1.4826
    X += rng.normal(0, 1, X.shape).astype(np.float32) * (s * mad)[:, None, None]
    return X

print(f"\n{'':>22} {'ac1':>6} {'bajas':>6} {'altas':>6}")
a = stats(Xd_c); print(f"{'DEMO (sin picos)':>22} {a[0]:6.3f} {a[1]:6.2f} {a[2]:6.2f}")
for s in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
    a = stats(add_noise(Xtr_c, s) if s else Xtr_c)
    print(f"{'train + ruido ' + str(s):>22} {a[0]:6.3f} {a[1]:6.2f} {a[2]:6.2f}")

def lmad(X):
    m = np.nanmedian(X, 2, keepdims=True)
    return np.log(np.nanmedian(np.abs(X - m), 2) * 1.4826 + 1e-3).mean(1)

lt, ld = lmad(Xtr), lmad(Xd)
print("\nlog-MAD por clase (media ± error típico)")
for k in range(8):
    a, b = lt[ytr == k], ld[yd == k]
    print(k, f"train {a.mean():.2f}±{a.std()/np.sqrt(len(a)):.2f}   demo {b.mean():.2f}±{b.std()/np.sqrt(len(b)):.2f}")