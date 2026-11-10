import time
import numpy as np
from competition_api import load_dataset
from model import Model

K, LAM = 8, 0.33
ds = load_dataset("data/test_demo.npz")
y = np.asarray(ds.labels)

m = Model()
for i in range(100):                      # calentamiento fuera del cálculo
    m.predict(ds[i % len(ds)])

pred, lat = [], []
for i in range(len(ds)):
    x = ds[i]                             # Sample ya construido, fuera del reloj
    t0 = time.perf_counter()
    p = m.predict(x)
    lat.append(time.perf_counter() - t0)
    pred.append(p)
pred, lat = np.array(pred), np.array(lat)

cm = np.zeros((K, K), int)
np.add.at(cm, (y, pred), 1)
n = cm.sum(1); c = np.diag(cm); w = n - c

print("Matriz de confusión (filas=real):\n", cm)
print("\nAcierto global:", (pred == y).mean().round(3))
print("Acierto por clase:", (c / n).round(2))
print(f"\nLatencia ms: media {lat.mean()*1e3:.3f} | mediana {np.median(lat)*1e3:.3f} | "
      f"p95 {np.percentile(lat,95)*1e3:.3f} | p99 {np.percentile(lat,99)*1e3:.3f} | max {lat.max()*1e3:.3f}")

# Extrapolación a una ejecución real: 60 s de tiempo en predict, clases equilibradas
N_total = 60 / lat.mean()
Nk = N_total / K
r = (c - LAM * w) / n
T = np.sqrt(Nk) * r * (c / n) ** 2
print(f"\nMuestras estimadas por ejecución: {N_total:,.0f} (~{Nk:,.0f} por clase)")
print("T_k por clase:", T.round(1))
print("PUNTUACIÓN ESTIMADA:", np.exp(np.log(np.maximum(0.01, T)).mean()).round(1))