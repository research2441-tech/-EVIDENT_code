"""Covariate-typicality scores for the typicality channel: for every frame and operating point
o=(k,r), the class-conditional Mahalanobis distances of the exit-k head embedding,
with class means and a tied covariance estimated on the training frames of the same seed.

usage: python features.py SEED
"""
import sys, numpy as np, torch, torch.nn.functional as F
from common import MultiExitNet, KNOWN, K2I, HELD_OUT, OPS, to_tensor, track_splits

seed = int(sys.argv[1]); D = "data"; OUT = "runs"
torch.set_num_threads(2)
g = np.load(f"{D}/gtsrb_tracks.npz")
X, Y, TID, FR, W = g["x"], g["y"], g["tid"], g["frame"], g["roi_w"]
tr, ca, te, unk = track_splits(TID, Y, seed)
net = MultiExitNet(); net.load_state_dict(torch.load(f"{OUT}/model_s{seed}.pt")); net.eval()

def emb(xu8, r):
    """head embeddings for exits 1..3 at resolution r: list of [N, d_k]"""
    outs = [[], [], []]
    with torch.no_grad():
        for i in range(0, len(xu8), 1024):
            x = to_tensor(np.ascontiguousarray(xu8[i:i + 1024]), r)
            f = net.s1(net.stem(x)); h = [f]
            f = net.s2(f); h.append(f); f = net.s3(f); h.append(f)
            for k, (hd, ff) in enumerate(zip((net.h1, net.h2, net.h3), h)):
                outs[k].append(F.adaptive_avg_pool2d(hd.p(ff), 1).flatten(1).numpy())
    return [np.concatenate(o) for o in outs]

itr = np.where(np.isin(TID, tr))[0]
ytr = np.array([K2I[c] for c in Y[itr]])
stats = {}
for r in (32, 48):
    E = emb(X[itr], r)
    for k in range(3):
        f = E[k].astype(np.float64)
        mu = np.stack([f[ytr == c].mean(0) for c in range(len(KNOWN))])
        Z = f - mu[ytr]; S = Z.T @ Z / len(Z) + 1e-3 * np.eye(f.shape[1])
        stats[(k + 1, r)] = (mu, np.linalg.inv(S))

def md(xu8):
    """class-conditional squared Mahalanobis distances [N, n_ops, K] (float16)"""
    out = np.zeros((len(xu8), len(OPS), len(KNOWN)), np.float16)
    for r in (32, 48):
        E = emb(xu8, r)
        for k in range(3):
            mu, P = stats[(k + 1, r)]
            L = np.linalg.cholesky(P)                       # P = L L^T  ->  d = ||L^T (f - mu)||^2
            fw = E[k].astype(np.float64) @ L; mw = mu @ L
            d = (fw ** 2).sum(1, keepdims=True) - 2 * fw @ mw.T + (mw ** 2).sum(1)[None]
            out[:, OPS.index((k + 1, r)), :] = np.maximum(d, 0).astype(np.float16)
    return out

def track_idx(tids):
    idx = []
    for t in tids:
        m = np.where(TID == t)[0]; idx += list(m[np.argsort(FR[m])])
    return np.array(idx)

tp = np.load(f"{D}/gtsrb_testpool.npz")
kn = np.where(np.isin(tp["y"], KNOWN))[0]
perm = np.random.default_rng(100 + seed).permutation(kn)
inorm = perm[int(0.3 * len(perm)):]
res = {}
for name, tids in (("cal", ca), ("test", te), ("unk", unk)):
    res[name] = md(X[track_idx(tids)])
res["norm"] = md(tp["x"][inorm])
it = track_idx(te)
for kind in ("night", "haze", "motion", "jpeg"):
    res[f"test_{kind}"] = md(np.load(f"{D}/shift_{kind}.npy", mmap_mode="r")[it])
for nm, fn in (("indian", "synth_indian"), ("synthctl", "synth_ctl")):
    res[nm] = md(np.load(f"{D}/{fn}.npz")["x"].reshape(-1, 48, 48, 3))
np.savez_compressed(f"{OUT}/typ_s{seed}.npz", **res)
print("done", seed, {k: v.shape for k, v in res.items()})
