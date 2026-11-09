# diag.py  (ejecútalo con el modelo que quieras analizar)
import sys
import numpy as np, torch
from cnn_common import Net, augment
from train_cnn import load, predict

ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else "cnn.pt", map_location="cpu")
net = Net(ck["width"]); net.load_state_dict(ck["state"]); net.eval()
use_meta = ck["use_meta"]

def per_class(y, p): return np.array([(p[y == k] == k).mean() for k in range(8)]).round(2)

# --- demo ---
Xd, Md, yd = load("data/test_demo.npz", use_meta)
pd = predict(net, Xd, Md, "cpu")
cm = np.zeros((8, 8), int); np.add.at(cm, (yd, pd), 1)
print("Matriz de confusión demo (filas=real, cols=predicho):\n", cm)

dev_ = np.nanmax(np.abs(Xd - np.nanmedian(Xd, 2, keepdims=True)), axis=(1, 2))
for name, m in [("sin pico >30", dev_ <= 30), ("con pico >30", dev_ > 30)]:
    print(f"{name}: n={m.sum()} acierto={np.mean(pd[m] == yd[m]):.3f}",
          "| clase 6:", f"{np.mean(pd[m & (yd == 6)] == 6):.2f}" if (m & (yd == 6)).any() else "-")

# --- validación del train (mismo split que train_cnn.py con seed 0) ---
X, M, y = load("data/train_100k.npz", use_meta)
vi = np.random.default_rng(0).permutation(len(y))[:5000]
pv = predict(net, X[vi], M[vi], "cpu")
pa = predict(net, augment(X[vi], np.random.default_rng(123)), M[vi], "cpu")
print("\nAcierto por clase, val limpio:", per_class(y[vi], pv))
print("Acierto por clase, val aug:   ", per_class(y[vi], pa))
print("Acierto por clase, demo:      ", per_class(yd, pd))