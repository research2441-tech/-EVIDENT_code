"""Pre-compute the deterministic test-time shifts once (shared by all seeds):
severity-3 corruptions of every GTSRB track frame, and synthetic approach tracks for the
Indian (cross-region) signs and for GTSRB test-pool control signs."""
import numpy as np
from common import corrupt, synth_track, KNOWN
D = "data"
g = np.load(f"{D}/gtsrb_tracks.npz")
X = g["x"]
for kind in ("night", "haze", "motion", "jpeg"):
    out = np.zeros_like(X)
    for i in range(0, len(X), 2000):
        out[i:i + 2000] = corrupt(X[i:i + 2000], kind, 3, np.random.default_rng(7 + i))
    np.save(f"{D}/shift_{kind}.npy", out); print(kind, flush=True)
ind = np.load(f"{D}/indian.npz")
srng = np.random.default_rng(2024)
sel = []
for c in np.unique(ind["cls"]):
    m = np.where(ind["cls"] == c)[0]; sel += list(srng.permutation(m)[:20])
sel = np.array(sorted(sel))
fr, ws = [], []
for i in sel:
    f, w = synth_track(ind["x"][i], srng); fr.append(f); ws.append(w)
np.savez_compressed(f"{D}/synth_indian.npz", x=np.stack(fr), w=np.stack(ws), y=ind["y"][sel], cls=ind["cls"][sel])
tp = np.load(f"{D}/gtsrb_testpool.npz")
kn = np.where(np.isin(tp["y"], KNOWN))[0]
crng = np.random.default_rng(2025)
ctl = crng.permutation(kn)[:500]
fr, ws = [], []
for i in ctl:
    f, w = synth_track(tp["x"][i], crng); fr.append(f); ws.append(w)
np.savez_compressed(f"{D}/synth_ctl.npz", x=np.stack(fr), w=np.stack(ws), y=tp["y"][ctl], idx=ctl)
print("indian tracks", len(sel), "control tracks", len(ctl))
