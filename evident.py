"""EVIDENT: energy-budgeted sequential evidence accumulation with track-level,
policy-agnostic time-uniform calibration for traffic-sign recognition on video tracks.

Two evidence channels are accumulated per tracked sign:
  label channel       A_t(y): rectified log calibrated e-values against every label y
  typicality channel  S_t   : signed log e-values of a label-free Mahalanobis typicality score
                       P_t   : its rectified (positive-part) version, used for flagging.
Both thresholds are calibrated on whole calibration tracks with a policy-agnostic envelope,
so the guarantees hold for any adaptive choice of frames and operating points.
All functions operate on pre-computed per-frame outputs so that every policy, baseline and
ablation is evaluated on exactly the same network outputs.
"""
import numpy as np
from common import OPS, KNOWN, group_of

K = len(KNOWN)
NOPS = len(OPS)
BIN_EDGES = np.array([25.0, 35.0, 50.0])          # apparent sign width bins (px)
FULL = OPS.index((3, 48))                          # most expensive operating point
ADMISSIBLE = [OPS.index(o) for o in [(2, 32), (3, 32), (2, 48), (3, 48)]]   # exits 2-3 at both inputs


def size_bin(w, edges=BIN_EDGES):
    return np.searchsorted(edges, w, side="right")


def _ref_lookup(ref, o, bb):
    if (o, bb) in ref:
        return ref[(o, bb)]
    cands = [k for k in ref if k[0] == o]
    return ref[min(cands, key=lambda k: abs(k[1] - bb))]


# ----------------------------------------------------------------------------- frame-level normalisation
class Normaliser:
    """Size- and operating-point-conditional conformal p-values for the label channel.
    p_o(x, y) = (1 + #{j in N_{o,b}: s_j >= s_o(x,y)}) / (|N_{o,b}| + 1),  s = -log pi_o(y|x)."""
    def __init__(self, lp, y, w, binned=True, edges=BIN_EDGES):
        self.binned, self.edges = binned, edges
        b = size_bin(w, edges) if binned else np.zeros(len(w), int)
        self.ref = {}
        for o in range(NOPS):
            s = -lp[np.arange(len(y)), o, y].astype(np.float64)
            for bb in np.unique(b):
                self.ref[(o, bb)] = np.sort(s[b == bb])

    def pvalues(self, lp, w):
        b = size_bin(w, self.edges) if self.binned else np.zeros(len(w), int)
        out = np.empty(lp.shape, np.float64)
        for o in range(NOPS):
            s = -lp[:, o, :].astype(np.float64)
            for bb in np.unique(b):
                m = b == bb; ref = _ref_lookup(self.ref, o, bb); n = len(ref)
                out[m, o, :] = (1.0 + n - np.searchsorted(ref, s[m], side="left")) / (n + 1.0)
        return out


class TypNormaliser:
    """Label-free typicality p-values u_o(x) from the minimum class-conditional Mahalanobis
    distance, referenced to normalisation frames of the same operating point and size bin."""
    def __init__(self, dist, w, edges=BIN_EDGES):
        self.edges = edges; b = size_bin(w, edges); sc = dist.astype(np.float64).min(2)
        self.ref = {(o, bb): np.sort(sc[b == bb, o]) for o in range(NOPS) for bb in np.unique(b)}

    def pvalues(self, dist, w):
        b = size_bin(w, self.edges); sc = dist.astype(np.float64).min(2); out = np.empty(sc.shape)
        for o in range(NOPS):
            for bb in np.unique(b):
                m = b == bb; ref = _ref_lookup(self.ref, o, bb); n = len(ref)
                out[m, o] = (1.0 + n - np.searchsorted(ref, sc[m, o], side="left")) / (n + 1.0)
        return out


def log_e(p, kappa):
    """log of the calibrated e-value kappa * p^(kappa-1)"""
    return np.log(kappa) + (kappa - 1.0) * np.log(p)


def log_evidence(p, kappa):
    """rectified log calibrated e-value"""
    return np.maximum(0.0, log_e(p, kappa))


def raw_evidence(lp, kappa):
    """ablation without conformal normalisation: the model NLL replaces -log p"""
    return np.maximum(0.0, np.log(kappa) + (1.0 - kappa) * (-lp.astype(np.float64)))


# ----------------------------------------------------------------------------- tracks and deadlines
class Tracks:
    def __init__(self, d, prefix, typ=None):
        self.lp = d[f"{prefix}__lp"]; self.y = d[f"{prefix}__y"]; self.w = d[f"{prefix}__w"]
        self.gid = d[f"{prefix}__gid"]; self.starts = d[f"{prefix}__starts"]
        self.n = len(self.starts) - 1
        self.ty = np.array([self.y[self.starts[i]] for i in range(self.n)])
        self.tgid = np.array([self.gid[self.starts[i]] for i in range(self.n)])
        self.grp = np.array([group_of(int(g)) for g in self.tgid])
        self.prefix = prefix
        self.dist = typ[prefix] if typ is not None else None
        if self.dist is not None:
            assert self.dist.shape == self.lp.shape, (prefix, self.dist.shape, self.lp.shape)
        if f"{prefix}__cls" in d:
            self.cls = d[f"{prefix}__cls"]

    def frames(self, i):
        return slice(self.starts[i], self.starts[i + 1])


def deadlines(tr, w_dl=(40.0, 50.0), min_frames=3):
    """number of usable frames: those observed before the apparent width reaches the
    group-specific deadline width (frames are ordered by increasing width)."""
    out = np.zeros(tr.n, int)
    for i in range(tr.n):
        w = tr.w[tr.frames(i)]
        out[i] = min(len(w), max(min_frames, int((w < w_dl[tr.grp[i]]).sum())))
    return out


def op_costs(table, mode="energy"):
    key = "energy_mJ" if mode == "energy" else "latency_ms"
    return np.array([table["ops"][f"{k}_{r}"][key] for (k, r) in OPS])


# ----------------------------------------------------------------------------- calibration
RHO = 32.0      # fixed resolution rule: 48-px input only when the crop is at least this wide


def frame_ops(w, ops):
    """admissible operating points for a frame of apparent width w under the fixed resolution
    rule (no up-sampling below RHO); if ops hold a single resolution, that resolution is used."""
    res = sorted({OPS[o][1] for o in ops})
    r = res[0] if len(res) == 1 else (48 if w >= RHO else 32)
    return [o for o in ops if OPS[o][1] == r]


def envelope_scores(tr, ev, dl, ops=None, label=True, free_res=False):
    """policy-agnostic track score R_i = sum_{t<dl} max_{o in O(w_t)} ell_{t,o}.
    label=True: ev[N, n_ops, K] evaluated at the true label; label=False: ev[N, n_ops]."""
    ops = ADMISSIBLE if ops is None else ops
    R = np.zeros(tr.n)
    for i in range(tr.n):
        f = tr.frames(i); W = tr.w[f]
        e = ev[f][: dl[i], :, tr.ty[i]] if label else ev[f][: dl[i]]
        R[i] = sum(e[t, ops if free_res else frame_ops(W[t], ops)].max() for t in range(dl[i]))
    return R


def conformal_threshold(scores, alpha):
    n = len(scores); k = int(np.ceil((1 - alpha) * (n + 1)))
    return np.inf if k > n else np.sort(scores)[k - 1]


def group_thresholds(R, grp, alphas):
    return np.array([conformal_threshold(R[grp == g], alphas[g]) for g in range(len(alphas))])


def grp_of_class_array():
    return np.array([group_of(c) for c in KNOWN])


GOC = grp_of_class_array()


# ----------------------------------------------------------------------------- EVIDENT (online)
def run_evident(tr, ev, p, dl, lam_group, costs, theta, lt=None, lam_typ=np.inf, a_conf=0.0,
                force_op=None, ops=None):
    """Simulate EVIDENT on every track.
    ev, p : label-channel evidence and p-values [N, n_ops, K]
    lt    : typicality-channel signed log e-values [N, n_ops] (None disables the channel)
    theta : (omega, gamma) = (minimum apparent width that receives computation,
             frame-level ambiguity level that triggers the next exit). The input resolution
             follows the fixed rule of frame_ops, so the envelope does not depend on theta.
    decisions: >=0 committed label, -1 unknown (empty set), -3 atypical flag, -2 deferred."""
    omega, gamma = theta[0], theta[1]
    ops = ADMISSIBLE if ops is None else ops
    lam_c = lam_group[GOC]
    n = tr.n
    dec = np.full(n, -2); t_dec = np.zeros(n, int); energy = np.zeros(n); nproc = np.zeros(n, int)
    true_out = np.zeros(n, bool); setsize = np.zeros(n)
    for i in range(n):
        f = tr.frames(i); E = ev[f]; P = p[f]; W = tr.w[f]; LT = lt[f] if lt is not None else None
        A = np.zeros(K); alive = np.ones(K, bool); S = 0.0; Pp = 0.0; en = 0.0; npr = 0
        for t in range(dl[i]):
            if force_op is not None:
                o = force_op; en += costs[o]
            else:
                if W[t] < omega and t < dl[i] - 1:
                    continue
                o_prev = None
                for o in sorted(frame_ops(W[t], ops), key=lambda q: OPS[q][0]):
                    en += costs[o] - (costs[o_prev] if o_prev is not None else 0.0); o_prev = o
                    if ((P[t, o, :] > gamma) & alive).sum() <= 1:
                        break
            npr += 1
            A += E[t, o, :]; alive = A <= lam_c
            if LT is not None:
                S += LT[t, o]; Pp += max(LT[t, o], 0.0)
                if Pp > lam_typ:
                    dec[i] = -3; t_dec[i] = t; break
            if alive.sum() == 0:
                dec[i] = -1; t_dec[i] = t; break
            if alive.sum() == 1 and S <= -a_conf:
                dec[i] = int(np.argmax(alive)); t_dec[i] = t; break
        else:
            t_dec[i] = dl[i] - 1
        energy[i] = en; nproc[i] = npr; setsize[i] = alive.sum()
        if tr.ty[i] >= 0:
            true_out[i] = not alive[tr.ty[i]]
    return dict(dec=dec, t=t_dec, energy=energy, nproc=nproc, setsize=setsize, true_out=true_out)


# ----------------------------------------------------------------------------- baselines (full operating point)
def base_single_frame(tr, dl, costs):
    n = tr.n; dec = np.array([int(np.argmax(tr.lp[tr.starts[i] + dl[i] - 1, FULL])) for i in range(n)])
    return dict(dec=dec, t=dl - 1, energy=np.full(n, costs[FULL]), nproc=np.ones(n, int))


def track_typicality(tr, dl, kind):
    """mean frame score over usable frames (for open-set thresholds of voting baselines)."""
    out = np.zeros(tr.n)
    for i in range(tr.n):
        f = tr.frames(i)
        if kind == "msp":
            out[i] = np.exp(tr.lp[f][: dl[i], FULL].astype(np.float64).max(1)).mean()
        else:
            out[i] = -tr.dist[f][: dl[i], FULL].astype(np.float64).min(1).mean()
    return out


def base_vote_hysteresis(tr, dl, costs, win=5, tau=3, share=0.6, exit_conf=None, reject=None):
    """confidence-weighted sliding-window voting with label locking (hysteresis).
    exit_conf: per-frame early exit at the first exit whose max probability >= exit_conf.
    reject: (kind, threshold) -> tracks whose mean typicality is below threshold are flagged."""
    n = tr.n; dec = np.full(n, -2); tt = np.zeros(n, int); en = np.zeros(n); npr = np.zeros(n, int)
    for i in range(n):
        L = tr.lp[tr.frames(i)].astype(np.float64); hist = []; run = 0; last = -1
        for t in range(dl[i]):
            if exit_conf is None:
                o = FULL; en[i] += costs[o]
            else:
                o_prev = None
                for k in (1, 2, 3):
                    o = OPS.index((k, 48)); en[i] += costs[o] - (costs[o_prev] if o_prev is not None else 0); o_prev = o
                    if np.exp(L[t, o].max()) >= exit_conf:
                        break
            npr[i] += 1
            pr = np.exp(L[t, o]); hist.append(pr)
            v = np.zeros(K)
            for q in hist[-win:]:
                v[np.argmax(q)] += q.max()
            lead = int(np.argmax(v)); sh = v[lead] / v.sum()
            run = run + 1 if (lead == last and sh >= share) else (1 if sh >= share else 0)
            last = lead
            if run >= tau:
                dec[i] = lead; tt[i] = t; break
        else:
            dec[i] = last; tt[i] = dl[i] - 1
    if reject is not None:
        kind, thr = reject
        dec[track_typicality(tr, dl, kind) < thr] = -3
    return dict(dec=dec, t=tt, energy=en, nproc=npr)


def base_frame_conformal(tr, p, dl, costs, alpha):
    """per-frame split conformal set at level alpha; commit at the first singleton set."""
    n = tr.n; dec = np.full(n, -2); tt = np.zeros(n, int); en = np.zeros(n); npr = np.zeros(n, int)
    for i in range(n):
        P = p[tr.frames(i)][:, FULL, :]
        for t in range(dl[i]):
            en[i] += costs[FULL]; npr[i] += 1
            s = P[t] > alpha
            if s.sum() == 1:
                dec[i] = int(np.argmax(s)); tt[i] = t; break
            if s.sum() == 0:
                dec[i] = -1; tt[i] = t; break
        else:
            tt[i] = dl[i] - 1
    return dict(dec=dec, t=tt, energy=en, nproc=npr)


def base_ville(tr, p, dl, costs, alpha, kappa):
    """product e-process per label, rejected when prod e >= 1/alpha (Ville's inequality);
    presumes conditionally valid frame e-values, which within-track dependence violates."""
    n = tr.n; dec = np.full(n, -2); tt = np.zeros(n, int); en = np.zeros(n); npr = np.zeros(n, int)
    thr = np.log(1 / alpha)
    for i in range(n):
        P = p[tr.frames(i)][:, FULL, :]; S = np.zeros(K)
        for t in range(dl[i]):
            en[i] += costs[FULL]; npr[i] += 1
            S += log_e(P[t], kappa); alive = S < thr
            if alive.sum() == 1:
                dec[i] = int(np.argmax(alive)); tt[i] = t; break
            if alive.sum() == 0:
                dec[i] = -1; tt[i] = t; break
        else:
            tt[i] = dl[i] - 1
    return dict(dec=dec, t=tt, energy=en, nproc=npr)


def base_bayes_sprt(tr, dl, costs, eta):
    """naive-Bayes posterior accumulation (frames treated as independent); commit when the
    leading posterior exceeds 1 - eta (eta tuned on calibration tracks)."""
    n = tr.n; dec = np.full(n, -2); tt = np.zeros(n, int); en = np.zeros(n); npr = np.zeros(n, int)
    for i in range(n):
        L = tr.lp[tr.frames(i)][:, FULL, :].astype(np.float64); S = np.zeros(K)
        for t in range(dl[i]):
            en[i] += costs[FULL]; npr[i] += 1
            S += L[t]; post = np.exp(S - S.max()); post /= post.sum()
            if post.max() >= 1 - eta:
                dec[i] = int(np.argmax(post)); tt[i] = t; break
        else:
            dec[i] = int(np.argmax(S)); tt[i] = dl[i] - 1
    return dict(dec=dec, t=tt, energy=en, nproc=npr)


# ----------------------------------------------------------------------------- metrics
def summarise(res, tr, dl):
    known = tr.ty >= 0; d = res["dec"]; out = {}
    if known.any():
        k = known
        out["committed_error"] = float(np.mean((d[k] >= 0) & (d[k] != tr.ty[k])))
        out["commit_correct"] = float(np.mean(d[k] == tr.ty[k]))
        out["defer"] = float(np.mean(d[k] == -2))
        out["abstain"] = float(np.mean((d[k] == -1) | (d[k] == -3)))
        out["energy_mJ"] = float(np.mean(res["energy"][k]))
        out["frames_processed"] = float(np.mean(res["nproc"][k]))
        c = d[k] >= 0
        out["lead_frames"] = float(np.mean((dl[k] - 1 - res["t"][k])[c])) if c.any() else 0.0
        com = d[k] >= 0
        out["energy_per_commit_mJ"] = float(res["energy"][k].sum() / max(com.sum(), 1))
        if "true_out" in res:
            out["true_label_excluded"] = float(np.mean(res["true_out"][k]))
    u = ~known
    if u.any():
        out["unk_abstain"] = float(np.mean((d[u] == -1) | (d[u] == -3)))
        out["unk_wrong_commit"] = float(np.mean(d[u] >= 0))
        out["unk_defer"] = float(np.mean(d[u] == -2))
        out["unk_energy_mJ"] = float(np.mean(res["energy"][u]))
    return out
