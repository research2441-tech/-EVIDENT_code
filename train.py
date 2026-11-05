"""Train the multi-exit backbone for one split seed and dump per-frame log-probabilities
for every operating point on every evaluation set.

usage: python train.py SEED [EPOCHS]
"""
import sys, time, json, numpy as np, torch, torch.nn.functional as F
from common import (MultiExitNet, KNOWN, K2I, HELD_OUT, OPS, to_tensor, track_splits, augment)

seed = int(sys.argv[1]); EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 14
D = "data"; OUT = "runs"
import os; os.makedirs(OUT, exist_ok=True)
torch.manual_seed(seed); np.random.seed(seed)
torch.set_num_threads(2)
rng = np.random.default_rng(seed)

g = np.load(f"{D}/gtsrb_tracks.npz")
X, Y, TID, FR, W = g["x"], g["y"], g["tid"], g["frame"], g["roi_w"]
tr, ca, te, unk = track_splits(TID, Y, seed)
itr = np.where(np.isin(TID, tr))[0]
ytr = np.array([K2I[c] for c in Y[itr]])

tp = np.load(f"{D}/gtsrb_testpool.npz")
kn = np.where(np.isin(tp["y"], KNOWN))[0]
perm = np.random.default_rng(100 + seed).permutation(kn)
ival, inorm = perm[: int(0.3 * len(perm))], perm[int(0.3 * len(perm)):]

net = MultiExitNet()
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=5e-4)
BS = 128
steps = EPOCHS * (len(itr) // BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.15)
WEX = (0.5, 0.75, 1.0)

def evaluate(idx_x, yy, res):
    net.eval(); acc = np.zeros(3); n = 0
    with torch.no_grad():
        for i in range(0, len(idx_x), 512):
            xb = to_tensor(idx_x[i:i + 512], res)
            outs = net(xb)
            for k in range(3):
                acc[k] += (outs[k].argmax(1).numpy() == yy[i:i + 512]).sum()
            n += len(xb)
    return acc / n

xval = tp["x"][ival]; yval = np.array([K2I[c] for c in tp["y"][ival]])
best, best_state, log = -1, None, []
t0 = time.time()
for ep in range(EPOCHS):
    net.train(); p = rng.permutation(len(itr)); tl = 0
    for b in range(len(itr) // BS):
        j = itr[p[b * BS:(b + 1) * BS]]
        xb = torch.from_numpy(X[j]).permute(0, 3, 1, 2).float() / 255.0
        xb = augment(xb, rng)
        res = int(rng.choice([32, 48]))
        if res != 48:
            xb = F.interpolate(xb, size=(res, res), mode="bilinear", antialias=True)
        from common import MEAN, STD
        xb = (xb - MEAN) / STD
        yb = torch.from_numpy(ytr[p[b * BS:(b + 1) * BS]])
        outs = net(xb)
        loss = sum(w * F.cross_entropy(o, yb) for w, o in zip(WEX, outs))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item()
    va = evaluate(xval, yval, 48)
    log.append({"epoch": ep, "loss": tl / (len(itr) // BS), "val_acc_exits_48": va.tolist(), "t": time.time() - t0})
    print(json.dumps(log[-1]), flush=True)
    if va[2] > best:
        best = va[2]; best_state = {k: v.clone() for k, v in net.state_dict().items()}
net.load_state_dict(best_state)
torch.save(best_state, f"{OUT}/model_s{seed}.pt")

# --------------------------------------------------------------------------- dump log-probs for all ops
def dump(xu8):
    """returns float16 array [N, n_ops, n_classes] of log-probabilities"""
    net.eval(); res_out = np.zeros((len(xu8), len(OPS), len(KNOWN)), np.float16)
    with torch.no_grad():
        for r in (32, 48):
            for i in range(0, len(xu8), 1024):
                outs = net(to_tensor(xu8[i:i + 1024], r))
                for oi, (k, rr) in enumerate(OPS):
                    if rr == r:
                        res_out[i:i + 1024, oi] = F.log_softmax(outs[k - 1].float(), 1).numpy()
    return res_out

def track_arrays(tids):
    idx, starts = [], []
    for t in tids:
        m = np.where(TID == t)[0]; m = m[np.argsort(FR[m])]
        starts.append(len(idx)); idx += list(m)
    return np.array(idx), np.array(starts + [len(idx)])

sets = {}
for name, tids in [("cal", ca), ("test", te), ("unk", unk)]:
    idx, st = track_arrays(tids)
    lab = np.array([K2I.get(int(c), -1) for c in Y[idx]])
    sets[name] = dict(lp=dump(X[idx]), y=lab, gid=Y[idx], w=W[idx], starts=st, tid=TID[idx])
sets["norm"] = dict(lp=dump(tp["x"][inorm]), y=np.array([K2I[c] for c in tp["y"][inorm]]), w=tp["roi_w"][inorm],
                    gid=tp["y"][inorm])
# corrupted test tracks (severity 3, pre-computed once by precompute_shift.py)
idx, st = track_arrays(te)
for kind in ("night", "haze", "motion", "jpeg"):
    xc = np.load(f"{D}/shift_{kind}.npy", mmap_mode="r")[idx]
    sets[f"test_{kind}"] = dict(lp=dump(np.ascontiguousarray(xc)), y=sets["test"]["y"], gid=sets["test"]["gid"],
                                w=W[idx], starts=st)
# synthetic approach tracks: Indian signs (cross-region) and GTSRB test-pool controls (pre-computed)
for nm, fn in (("indian", "synth_indian"), ("synthctl", "synth_ctl")):
    s_ = np.load(f"{D}/{fn}.npz")
    n, T = s_["x"].shape[:2]
    gid = np.repeat(s_["y"], T)
    sets[nm] = dict(lp=dump(s_["x"].reshape(-1, 48, 48, 3)), y=np.array([K2I.get(int(c), -1) for c in gid]),
                    gid=gid, w=s_["w"].reshape(-1), starts=np.arange(0, n * T + 1, T))
    if "cls" in s_:
        sets[nm]["cls"] = s_["cls"]
# held-out class images in the test pool (frame-level open-set check)
hu = np.where(np.isin(tp["y"], HELD_OUT))[0]
sets["unkframes"] = dict(lp=dump(tp["x"][hu]), y=-np.ones(len(hu), int), gid=tp["y"][hu], w=tp["roi_w"][hu])

flat = {}
for s, d in sets.items():
    for k, v in d.items():
        flat[f"{s}__{k}"] = v
np.savez_compressed(f"{OUT}/outputs_s{seed}.npz", **flat)
json.dump({"log": log, "best_val_final_exit": float(best), "n_train_tracks": len(tr), "n_cal_tracks": len(ca),
           "n_test_tracks": len(te), "n_unk_tracks": len(unk), "n_train_frames": int(len(itr)),
           "n_norm": int(len(inorm)), "n_val": int(len(ival)), "train_time_s": time.time() - t0},
          open(f"{OUT}/train_s{seed}.json", "w"), indent=1)
print("done seed", seed, "best", best, "time", time.time() - t0)
