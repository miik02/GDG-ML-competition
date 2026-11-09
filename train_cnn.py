import argparse
import numpy as np
import torch
import torch.nn.functional as F
from competition_api import load_dataset
from cnn_common import GROUPS, Net, augment, build_input


def load(path, use_meta):
    ds = load_dataset(path)
    X = np.array(ds.series, dtype=np.float32)
    sr = (np.asarray(ds.meta["sample_rate"]) >= 1.5).astype(np.float32)
    g = np.asarray(ds.meta["group"]).astype(str)
    M = np.stack([sr] + [(g == k).astype(np.float32) for k in GROUPS], 1)
    return X, (M if use_meta else M * 0), np.asarray(ds.labels).astype(np.int64)


def per_class(y, p):
    return np.array([(p[y == k] == k).mean() for k in range(8)])


def proxy_score(y, p, lam=0.33):
    """Puntuación sin el factor √n (igual para modelos igual de rápidos)."""
    t = []
    for k in range(8):
        n = (y == k).sum(); c = (p[y == k] == k).sum()
        t.append(max(0.01, (c - lam * (n - c)) / n * (c / n) ** 2))
    return float(np.exp(np.mean(np.log(t))))


@torch.no_grad()
def predict(net, X, M, dev, bs=1000):
    net.eval()
    out = [net(torch.from_numpy(build_input(X[i:i + bs], M[i:i + bs])).to(dev)).argmax(1).cpu()
           for i in range(0, len(X), bs)]
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train_100k.npz")
    ap.add_argument("--demo", default="data/test_demo.npz")
    ap.add_argument("--out", default="cnn.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    use_meta = not a.no_meta
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, M, y = load(a.train, use_meta)
    Xd, Md, yd = load(a.demo, use_meta)

    perm = rng.permutation(len(y)); vi, ti = perm[:5000], perm[5000:]
    Xva_aug = augment(X[vi], np.random.default_rng(123))  # validación "con el test simulado"

    net = Net(a.width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-2)
    nb = len(ti) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * nb)

    for ep in range(a.epochs):
        net.train(); idx = rng.permutation(ti); tot = 0.0
        for i in range(nb):
            b = idx[i * a.bs:(i + 1) * a.bs]
            inp = torch.from_numpy(build_input(augment(X[b], rng), M[b])).to(dev)
            loss = F.cross_entropy(net(inp), torch.from_numpy(y[b]).to(dev), label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        pv = predict(net, X[vi], M[vi], dev); pa = predict(net, Xva_aug, M[vi], dev); pd = predict(net, Xd, Md, dev)
        print(f"ep {ep+1:2d} loss {tot/nb:.3f} | val limpio {np.mean(pv==y[vi]):.3f} | val aug {np.mean(pa==y[vi]):.3f} "
              f"| demo acc {np.mean(pd==yd):.3f} proxy {proxy_score(yd, pd):.3f}")

    print("acierto por clase en demo:", per_class(yd, pd).round(2))
    torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()}, "width": a.width, "use_meta": use_meta}, a.out)
    print("guardado", a.out)


if __name__ == "__main__":
    main()