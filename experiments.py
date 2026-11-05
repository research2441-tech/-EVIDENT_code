"""Run every experiment on the dumped per-frame outputs of all seeds.
Writes runs/<tag>.json (summaries) and runs/pertrack_s{seed}.npz (per-track records).
usage: python experiments.py 0,1,2,3,4 results
"""
import json, sys, itertools, time, numpy as np
from evident import *

SEEDS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2, 3, 4]
PROF = json.load(open("runs/energy_profile.json"))
COST_E = op_costs(PROF, "energy"); COST_L = op_costs(PROF, "latency")
ALPHA = np.array([0.02, 0.05])        # label channel: regulatory, warning/other
ALPHA_T = 0.05                        # typicality channel: false atypical flags on known signs
KAPPA = 0.02; KAPPA_T = 0.1; A_CONF = 2.0
W_DL = (40.0, 50.0)
DELTA = 0.02                          # allowed rise in non-commitment over dense processing (calibration tracks)
GRID = list(itertools.product([0, 16, 18, 20, 22, 24, 26, 28, 30, 34], [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]))
SETS = ["test", "unk", "test_night", "test_haze", "test_motion", "test_jpeg", "indian", "synthctl"]


class Ctx:
    """everything derived from one seed's outputs for one configuration"""
    def __init__(self, d, typ, norm, tnorm, name, kappa=KAPPA, kappa_t=KAPPA_T, raw=False, w_dl=W_DL):
        self.tr = tr = Tracks(d, name, typ)
        self.p = norm.pvalues(tr.lp, tr.w)
        self.ev = raw_evidence(tr.lp, kappa) if raw else log_evidence(self.p, kappa)
        self.u = tnorm.pvalues(tr.dist, tr.w)
        self.lt = log_e(self.u, kappa_t)
        self.dl = deadlines(tr, w_dl)


def calibrate(c, ops=None, alpha=ALPHA, alpha_t=ALPHA_T):
    R = envelope_scores(c.tr, c.ev, c.dl, ops)
    lam = group_thresholds(R, c.tr.grp, alpha)
    Rt = envelope_scores(c.tr, np.maximum(c.lt, 0), c.dl, ops, label=False)
    return lam, conformal_threshold(Rt, alpha_t), R, Rt


def lagrangian_select(c, lam, lam_t, costs, delta=DELTA, grid=GRID, typ=True, ops=None):
    """dual bisection: theta*(mu) = argmin E[energy] + mu * P(no commit), smallest mu whose
    non-commitment does not exceed that of dense processing (omega=0, smallest gamma) by more than delta."""
    rows = []
    for th in grid:
        r = run_evident(c.tr, c.ev, c.p, c.dl, lam, costs, th, c.lt if typ else None, lam_t, A_CONF if typ else 0, ops=ops)
        rows.append((th, r["energy"].mean(), np.mean(r["dec"] < 0)))
    E = np.array([r[1] for r in rows]); Dn = np.array([r[2] for r in rows])
    gmin = min(g[1] for g in grid)
    dense = [i for i, r in enumerate(rows) if r[0][0] == 0 and r[0][1] == gmin][0]
    beta = Dn[dense] + delta
    pick = lambda mu: int(np.argmin(E + mu * Dn))
    if Dn[pick(1e5)] > beta:
        return rows[int(np.argmin(Dn))][0], rows, None, beta
    lo, hi = 0.0, 1e5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if Dn[pick(mid)] <= beta:
            hi = mid
        else:
            lo = mid
    return rows[pick(hi)][0], rows, hi, beta


def evident(c, lam, lam_t, th, costs=COST_E, typ=True, force=None, ops=None):
    return run_evident(c.tr, c.ev, c.p, c.dl, lam, costs, th, c.lt if typ else None, lam_t,
                       A_CONF if typ else 0.0, force_op=force, ops=ops)


def seed_run(seed):
    t0 = time.time()
    d = np.load(f"runs/outputs_s{seed}.npz", allow_pickle=True)
    typ = np.load(f"runs/typ_s{seed}.npz")
    norm = Normaliser(d["norm__lp"], d["norm__y"], d["norm__w"])
    tnorm = TypNormaliser(typ["norm"], d["norm__w"])
    out = {"seed": seed}
    C = {n: Ctx(d, typ, norm, tnorm, n) for n in ["cal"] + SETS if f"{n}__lp" in d}
    cal = C["cal"]
    lam, lam_t, R, Rt = calibrate(cal)
    th, grid_rows, mu, beta = lagrangian_select(cal, lam, lam_t, COST_E)
    lam_s, lam_ts, _, _ = calibrate(cal, ops=[FULL])
    lam_l, _, _, _ = calibrate(cal)
    th_l, _, _, _ = lagrangian_select(cal, lam_l, np.inf, COST_E, typ=False)
    out.update(lam=lam.tolist(), lam_t=float(lam_t), theta=list(th), mu=mu, beta=float(beta), lam_static=lam_s.tolist(),
               lam_t_static=float(lam_ts), theta_labelonly=list(th_l), n_cal=int(cal.tr.n),
               n_cal_groups=[int((cal.tr.grp == g).sum()) for g in (0, 1)],
               grid=[[list(r[0]), float(r[1]), float(r[2])] for r in grid_rows])
    # tuned baselines on calibration tracks
    etas = np.logspace(-10, -0.5, 40); best = etas[0]; k = cal.tr.ty >= 0
    for e in etas:
        r = base_bayes_sprt(cal.tr, cal.dl, COST_E, e)
        if np.mean((r["dec"][k] >= 0) & (r["dec"][k] != cal.tr.ty[k])) <= 0.9 * ALPHA.mean():
            best = e
    out["eta_sprt"] = float(best)
    thr_msp = float(np.quantile(track_typicality(cal.tr, cal.dl, "msp"), 0.05))
    thr_md = float(np.quantile(track_typicality(cal.tr, cal.dl, "md"), 0.05))
    out.update(thr_msp=thr_msp, thr_md=thr_md)

    pert = {}; out["sets"] = {}
    for name in SETS:
        if name not in C:
            continue
        c = C[name]; tr, dl = c.tr, c.dl
        meth = {
            "EVIDENT": evident(c, lam, lam_t, th),
            "EVIDENT-static": evident(c, lam_s, lam_ts, th, force=FULL, ops=[FULL]),
            "EVIDENT-label-only": evident(c, lam_l, np.inf, th_l, typ=False),
            "Single-frame@deadline": base_single_frame(tr, dl, COST_E),
            "Vote+hysteresis": base_vote_hysteresis(tr, dl, COST_E),
            "Vote+hysteresis+MSP": base_vote_hysteresis(tr, dl, COST_E, reject=("msp", thr_msp)),
            "Vote+hysteresis+Maha": base_vote_hysteresis(tr, dl, COST_E, reject=("md", thr_md)),
            "EarlyExit+vote": base_vote_hysteresis(tr, dl, COST_E, exit_conf=0.9),
            "Frame-CP first-singleton": base_frame_conformal(tr, c.p, dl, COST_E, ALPHA.mean()),
            "E-process (Ville)": base_ville(tr, c.p, dl, COST_E, ALPHA.mean(), KAPPA),
            "Bayes-SPRT (tuned)": base_bayes_sprt(tr, dl, COST_E, best),
        }
        out["sets"][name] = {m: summarise(r, tr, dl) for m, r in meth.items()}
        out["sets"][name]["_n_tracks"] = int(tr.n); out["sets"][name]["_mean_deadline_frames"] = float(dl.mean())
        if name in ("test", "unk", "test_night", "indian"):
            for m, r in meth.items():
                for k2 in ("dec", "t", "energy", "nproc"):
                    pert[f"{name}|{m}|{k2}"] = r[k2]
            pert[f"{name}|ty"] = tr.ty; pert[f"{name}|grp"] = tr.grp; pert[f"{name}|dl"] = dl
            pert[f"{name}|tgid"] = tr.tgid
            r = meth["EVIDENT"]
            pert[f"{name}|EVIDENT|wdec"] = np.array([tr.w[tr.starts[i] + r["t"][i]] for i in range(tr.n)])
            pert[f"{name}|wdl"] = np.array([tr.w[tr.starts[i] + dl[i] - 1] for i in range(tr.n)])
        if name == "indian":
            r = meth["EVIDENT"]; rv = meth["Vote+hysteresis"]; out["indian_per_class"] = {}
            for cl in np.unique(tr.cls):
                m = tr.cls == cl; kn = bool(tr.ty[m][0] >= 0)
                out["indian_per_class"][str(cl)] = {
                    "known": kn, "n": int(m.sum()),
                    "correct": float(np.mean(r["dec"][m] == tr.ty[m])) if kn else None,
                    "wrong_commit": float(np.mean((r["dec"][m] >= 0) & (r["dec"][m] != tr.ty[m]))),
                    "abstain": float(np.mean(r["dec"][m] <= -1) - np.mean(r["dec"][m] == -2)),
                    "defer": float(np.mean(r["dec"][m] == -2)),
                    "vote_wrong": float(np.mean((rv["dec"][m] >= 0) & (rv["dec"][m] != tr.ty[m])))}
    test = C["test"]
    pert["test|R"] = envelope_scores(test.tr, test.ev, test.dl)
    pert["test|Rt"] = envelope_scores(test.tr, np.maximum(test.lt, 0), test.dl, label=False)
    pert["cal|R"] = R; pert["cal|Rt"] = Rt; pert["cal|grp"] = cal.tr.grp; pert["test|grp2"] = test.tr.grp
    # example evidence trajectories (Fig.)
    out["timing_main_s"] = time.time() - t0

    # ---------------- E2: factorial ablation (A calibration, B normalisation, C controller, D typicality)
    rawc = Ctx(d, typ, norm, tnorm, "cal", raw=True); rawt = Ctx(d, typ, norm, tnorm, "test", raw=True)
    abl = {}
    for A_on, B_on, C_on, D_on in itertools.product([0, 1], [0, 1], [0, 1], [0, 1]):
        cc, ct = (cal, test) if B_on else (rawc, rawt)
        ops = None if C_on else [FULL]
        if A_on:
            lm, lmt, _, _ = calibrate(cc, ops=ops)
        else:
            lm, lmt = np.log(1 / ALPHA), np.log(1 / ALPHA_T)
        if C_on:
            th_ab, _, _, _ = lagrangian_select(cc, lm, lmt, COST_E, typ=bool(D_on))
            r = evident(ct, lm, lmt, th_ab, typ=bool(D_on))
        else:
            r = evident(ct, lm, lmt, th, typ=bool(D_on), force=FULL, ops=[FULL])
        s = summarise(r, ct.tr, ct.dl)
        for nm in ("unk", "test_night"):
            cu = C[nm] if B_on else Ctx(d, typ, norm, tnorm, nm, raw=True)
            ru = evident(cu, lm, lmt, th_ab if C_on else th, typ=bool(D_on), force=None if C_on else FULL,
                         ops=None if C_on else [FULL])
            su = summarise(ru, cu.tr, cu.dl)
            s[f"{nm}_wrong"] = su.get("unk_wrong_commit", su.get("committed_error"))
        abl[f"A{A_on}B{B_on}C{C_on}D{D_on}"] = s
    out["ablation"] = abl

    # ---------------- E3: sensitivity
    sens = {}
    for a in (0.01, 0.02, 0.05, 0.1, 0.2):
        al = np.array([a, a]); lm = group_thresholds(R, cal.tr.grp, al)
        rs = evident(test, lm, lam_t, th)
        rl = evident(test, group_thresholds(R, cal.tr.grp, al), np.inf, th_l, typ=False)
        fc = base_frame_conformal(test.tr, test.p, test.dl, COST_E, a)
        vl = base_ville(test.tr, test.p, test.dl, COST_E, a, KAPPA)
        sens[f"alpha={a}"] = {"EVIDENT": summarise(rs, test.tr, test.dl), "EVIDENT-label-only": summarise(rl, test.tr, test.dl),
                              "Frame-CP": summarise(fc, test.tr, test.dl), "Ville": summarise(vl, test.tr, test.dl)}
    for kp in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        cc = Ctx(d, typ, norm, tnorm, "cal", kappa=kp); ct = Ctx(d, typ, norm, tnorm, "test", kappa=kp)
        lm, lmt, _, _ = calibrate(cc); thk, _, _, _ = lagrangian_select(cc, lm, lmt, COST_E)
        sens[f"kappa={kp}"] = summarise(evident(ct, lm, lmt, thk), ct.tr, ct.dl)
    for ac in (0.0, 1.0, 2.0, 4.0, 6.0):
        r1 = run_evident(test.tr, test.ev, test.p, test.dl, lam, COST_E, th, test.lt, lam_t, ac)
        cu = C["unk"]; r2 = run_evident(cu.tr, cu.ev, cu.p, cu.dl, lam, COST_E, th, cu.lt, lam_t, ac)
        cn = C["test_night"]; r3 = run_evident(cn.tr, cn.ev, cn.p, cn.dl, lam, COST_E, th, cn.lt, lam_t, ac)
        sens[f"a_conf={ac}"] = {"test": summarise(r1, test.tr, test.dl), "unk": summarise(r2, cu.tr, cu.dl),
                                "night": summarise(r3, cn.tr, cn.dl)}
    for opsname, ops in (("all6", list(range(6))), ("exits23", ADMISSIBLE), ("exit3", [OPS.index((3, 32)), FULL]),
                         ("exits23_48only", [OPS.index((2, 48)), FULL])):
        lm, lmt, _, _ = calibrate(cal, ops=ops); tho, _, _, _ = lagrangian_select(cal, lm, lmt, COST_E, ops=ops)
        sens[f"ops={opsname}"] = {"lam": lm.tolist(), "lam_t": float(lmt), **summarise(evident(test, lm, lmt, tho, ops=ops), test.tr, test.dl)}
    for sc in (0.8, 1.0, 1.25, 1.5):
        wd = (W_DL[0] * sc, W_DL[1] * sc)
        cc = Ctx(d, typ, norm, tnorm, "cal", w_dl=wd); ct = Ctx(d, typ, norm, tnorm, "test", w_dl=wd)
        lm, lmt, _, _ = calibrate(cc); th2, _, _, _ = lagrangian_select(cc, lm, lmt, COST_E)
        sens[f"deadline_scale={sc}"] = {"EVIDENT": summarise(evident(ct, lm, lmt, th2), ct.tr, ct.dl),
                                        "Vote+hysteresis": summarise(base_vote_hysteresis(ct.tr, ct.dl, COST_E), ct.tr, ct.dl),
                                        "Single-frame": summarise(base_single_frame(ct.tr, ct.dl, COST_E), ct.tr, ct.dl),
                                        "mean_deadline_frames": float(ct.dl.mean())}
    for b in (0.0, 0.01, 0.02, 0.05, 0.1):
        th3, _, _, _ = lagrangian_select(cal, lam, lam_t, COST_E, delta=b)
        sens[f"delta={b}"] = {"theta": list(th3), **summarise(evident(test, lam, lam_t, th3), test.tr, test.dl)}
    th4, _, _, _ = lagrangian_select(cal, lam, lam_t, COST_L)
    sens["cost=latency"] = {"theta": list(th4),
                            "EVIDENT": summarise(evident(test, lam, lam_t, th4, costs=COST_L), test.tr, test.dl),
                            "EVIDENT-static": summarise(evident(test, lam_s, lam_ts, th4, costs=COST_L, force=FULL, ops=[FULL]), test.tr, test.dl),
                            "Vote+hysteresis": summarise(base_vote_hysteresis(test.tr, test.dl, COST_L), test.tr, test.dl),
                            "EarlyExit+vote": summarise(base_vote_hysteresis(test.tr, test.dl, COST_L, exit_conf=0.9), test.tr, test.dl)}
    nu = Normaliser(d["norm__lp"], d["norm__y"], d["norm__w"], binned=False)
    cu_ = Ctx(d, typ, nu, tnorm, "cal"); tu_ = Ctx(d, typ, nu, tnorm, "test")
    lm, lmt, _, _ = calibrate(cu_); thu, _, _, _ = lagrangian_select(cu_, lm, lmt, COST_E)
    sens["unbinned_normaliser"] = summarise(evident(tu_, lm, lmt, thu), tu_.tr, tu_.dl)
    # thresholds if the resolution were also left to the controller (envelope over both inputs per frame)
    Rm = envelope_scores(cal.tr, cal.ev, cal.dl, ADMISSIBLE, free_res=True)
    sens["lam_free_resolution"] = group_thresholds(Rm, cal.tr.grp, ALPHA).tolist()
    out["sensitivity"] = sens

    # ---------------- E7: re-split validity distribution (no retraining)
    rng = np.random.default_rng(1000 + seed)
    pool_R = np.concatenate([R, pert["test|R"]]); pool_Rt = np.concatenate([Rt, pert["test|Rt"]])
    pool_g = np.concatenate([cal.tr.grp, test.tr.grp]); ncal = cal.tr.n
    viol, viol_g, flag = [], [], []
    for rep in range(500):
        pm = rng.permutation(len(pool_R)); ci, ti = pm[:ncal], pm[ncal:]
        lm = group_thresholds(pool_R[ci], pool_g[ci], ALPHA); v = pool_R[ti] > lm[pool_g[ti]]
        viol.append(v.mean()); viol_g.append([v[pool_g[ti] == g].mean() for g in (0, 1)])
        flag.append(np.mean(pool_Rt[ti] > conformal_threshold(pool_Rt[ci], ALPHA_T)))
    out["resplit"] = {"viol": viol, "viol_g": viol_g, "flag": flag}
    # calibration-set size
    nc = {}
    for m_ in (60, 100, 150, 200):
        errs, ens, ncm = [], [], []
        for rep in range(30):
            sub = rng.choice(ncal, m_, replace=False)
            lm = group_thresholds(R[sub], cal.tr.grp[sub], ALPHA); lmt = conformal_threshold(Rt[sub], ALPHA_T)
            s = summarise(evident(test, lm, lmt, th), test.tr, test.dl)
            errs.append(s["committed_error"]); ens.append(s["energy_mJ"]); ncm.append(1 - s["commit_correct"] - s["committed_error"])
        nc[m_] = {"err_mean": float(np.mean(errs)), "err_q90": float(np.quantile(errs, 0.9)),
                  "energy": float(np.mean(ens)), "noncommit": float(np.mean(ncm))}
    out["ncal"] = nc
    # evidence trajectories for one regulatory test track (figure)
    np.savez_compressed(f"runs/pertrack_s{seed}.npz", **pert)
    out["timing_total_s"] = time.time() - t0
    return out


if __name__ == "__main__":
    allr = []
    tag = sys.argv[2] if len(sys.argv) > 2 else "results"
    for s in SEEDS:
        r = seed_run(s); allr.append(r)
        print("seed", s, "theta", r["theta"], "lam", np.round(r["lam"], 2), "lam_t", round(r["lam_t"], 2),
              json.dumps({k: round(v, 4) for k, v in r["sets"]["test"]["EVIDENT"].items()}), "time", round(r["timing_total_s"]), flush=True)
        json.dump(allr, open(f"runs/{tag}.json", "w"))
