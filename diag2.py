import numpy as np, torch
from competition_api import load_dataset
from cnn_common import Net, normalize
from train_cnn import load, predict

tr = load_dataset("data/train_100k.npz"); te = load_dataset("data/test_demo.npz")
ytr = np.asarray(tr.labels); yte = np.asarray(te.labels)
Xtr = np.asarray(tr.series[:30000]); Xte = np.asarray(te.series)

def logmad(X):
    med = np.nanmedian(X, 2, keepdims=True)
    return np.log(np.nanmedian(np.abs(X - med), 2) * 1.4826 + 1e-3).mean(1)

a, b = logmad(Xtr), logmad(Xte)
print("log-MAD por clase  train:", [round(float(a[ytr[:30000] == k].mean()), 2) for k in range(8)])
print("log-MAD por clase  demo :", [round(float(b[yte == k].mean()), 2) for k in range(8)])

def bands(X):  # energía por banda de frecuencia de la señal normalizada
    z = normalize(X)[:, :4]
    p = (np.abs(np.fft.rfft(z, axis=2)) ** 2)[:, :, 1:].mean(1)
    return np.log10(np.stack([c.sum(1) for c in np.array_split(p, 8, axis=1)], 1) + 1e-9)

bt, bd = bands(Xtr), bands(Xte)
print("\nbandas demo - train, por clase (filas=clase, cols=8 bandas de baja a alta frecuencia):")
for k in range(8):
    print(k, (bd[yte == k].mean(0) - bt[ytr[:30000] == k].mean(0)).round(2))

# acierto por group y sample_rate: demo vs validación
ck = torch.load("cnn_nometa.pt", map_location="cpu")
net = Net(ck["width"]); net.load_state_dict(ck["state"]); net.eval()
Xd, Md, yd = load("data/test_demo.npz", ck["use_meta"]); pd = predict(net, Xd, Md, "cpu")
X, M, y = load("data/train_100k.npz", ck["use_meta"])
vi = np.random.default_rng(0).permutation(len(y))[:5000]; pv = predict(net, X[vi], M[vi], "cpu")
for name, key in [("group", "group"), ("sample_rate", "sample_rate")]:
    gd, gv = np.asarray(te.meta[key]), np.asarray(tr.meta[key])[vi]
    for v in np.unique(gd):
        print(f"{name}={v}: demo {np.mean(pd[gd == v] == yd[gd == v]):.2f}  val {np.mean(pv[gv == v] == y[vi][gv == v]):.2f}")