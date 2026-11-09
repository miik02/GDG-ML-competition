import numpy as np
from competition_api import load_dataset
from cnn_common import normalize

tr = load_dataset("data/train_100k.npz"); te = load_dataset("data/test_demo.npz")
X, y = np.asarray(tr.series), np.asarray(tr.labels)
Xd, yd = np.asarray(te.series), np.asarray(te.labels)
idx = np.random.default_rng(0).permutation(len(y))[:30000]   # ALEATORIO, el fichero está ordenado por clase
X, y = X[idx], y[idx]

def feats(X):
    med = np.nanmedian(X, 2, keepdims=True)
    lm = np.log(np.nanmedian(np.abs(X - med), 2) * 1.4826 + 1e-3).mean(1)
    z = normalize(X)[:, :4]
    ac1 = (z[:, :, 1:] * z[:, :, :-1]).mean((1, 2)) / (z.var((1, 2)) + 1e-9)
    p = (np.abs(np.fft.rfft(z, axis=2)) ** 2).mean(1)[:, 1:]
    spec = np.log10(np.stack([c.sum(1) for c in np.array_split(p, 16, axis=1)], 1) + 1e-9)
    return lm, ac1, spec

lt, at, st = feats(X); ld, ad, sd = feats(Xd)
print("clase | logMAD train/demo | autocorr lag1 train/demo")
for k in range(8):
    print(k, round(lt[y == k].mean(), 2), round(ld[yd == k].mean(), 2), "|",
          round(at[y == k].mean(), 2), round(ad[yd == k].mean(), 2))

print("\nespectro (16 bandas, baja->alta), log10")
for name, S, lab, k in [("train 6", st, y, 6), ("train 7", st, y, 7), ("demo  6", sd, yd, 6), ("demo  7", sd, yd, 7)]:
    print(name, S[lab == k].mean(0).round(2))
print("\ndemo - train, por clase:")
for k in range(8):
    print(k, (sd[yd == k].mean(0) - st[y == k].mean(0)).round(2))