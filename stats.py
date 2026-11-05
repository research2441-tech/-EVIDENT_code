"""Aggregate all seeds: means, SDs, pooled per-track tests. Writes runs/stats.json"""
import json, numpy as np
from scipy import stats as st

R = json.load(open("runs/results.json"))
SEEDS = [r["seed"] for r in R]
METHODS = list(R[0]["sets"]["test"].keys()); METHODS = [m for m in METHODS if not m.startswith("_")]
BASE = [m for m in METHODS if m != "EVIDENT"]


def ms(path):
    v = []
    for r in R:
        x = r
        for p in path:
            x = x[p]
        v.append(x)
    v = np.array(v, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def pooled(setname, key):
    out = {}
    for m in METHODS:
        out[m] = np.concatenate([np.load(f"runs/pertrack_s{s}.npz")[f"{setname}|{m}|{key}"] for s in SEEDS])
    return out


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for rank, i in enumerate(o):
        run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def cp_upper(k, n, conf=0.95):
    return 1.0 if k == n else float(st.beta.ppf(conf, k + 1, n - k))


S = {"seeds": SEEDS, "methods": METHODS}
# ------------------------------------------------ summary tables
for setname in R[0]["sets"]:
    S[setname] = {m: {k: ms(("sets", setname, m, k)) for k in R[0]["sets"][setname][m]} for m in METHODS}
S["ablation"] = {k: {kk: ms(("ablation", k, kk)) for kk in R[0]["ablation"][k]} for k in R[0]["ablation"]}
sens = {}
for k, v in R[0]["sensitivity"].items():
    if isinstance(v, list):
        sens[k] = [ms(("sensitivity", k, j)) for j in range(len(v))]
        continue
    if all(isinstance(x, dict) for x in v.values() if not isinstance(x, (list, float, int))) and any(isinstance(x, dict) for x in v.values()):
        sens[k] = {}
        for kk, vv in v.items():
            if isinstance(vv, dict):
                sens[k][kk] = {q: ms(("sensitivity", k, kk, q)) for q in vv if isinstance(vv[q], (int, float))}
            elif isinstance(vv, (int, float)):
                sens[k][kk] = ms(("sensitivity", k, kk))
    else:
        sens[k] = {q: ms(("sensitivity", k, q)) for q in v if isinstance(v[q], (int, float))}
S["sensitivity"] = sens
S["lam"] = [ms(("lam", 0)), ms(("lam", 1))]; S["lam_t"] = ms(("lam_t",))
S["lam_static"] = [ms(("lam_static", 0)), ms(("lam_static", 1))]
S["theta"] = [r["theta"] for r in R]; S["beta"] = [r["beta"] for r in R]; S["n_cal_groups"] = [r["n_cal_groups"] for r in R]
S["ncal"] = {k: {q: ms(("ncal", k, q)) for q in R[0]["ncal"][k]} for k in R[0]["ncal"]}
vg = np.concatenate([np.array(r["resplit"]["viol_g"]) for r in R]); fl = np.concatenate([np.array(r["resplit"]["flag"]) for r in R])
S["resplit"] = {"n": int(len(vg)), "reg_mean": float(vg[:, 0].mean()), "warn_mean": float(vg[:, 1].mean()),
                "flag_mean": float(fl.mean()), "reg_q95": float(np.quantile(vg[:, 0], 0.95)),
                "warn_q95": float(np.quantile(vg[:, 1], 0.95)), "flag_q95": float(np.quantile(fl, 0.95))}

# ------------------------------------------------ pooled per-track tests
tests = {}
for setname in ("test", "test_night"):
    E = pooled(setname, "energy"); D = pooled(setname, "dec")
    ty = np.concatenate([np.load(f"runs/pertrack_s{s}.npz")[f"{setname}|ty"] for s in SEEDS])
    grp = np.concatenate([np.load(f"runs/pertrack_s{s}.npz")[f"{setname}|grp"] for s in SEEDS])
    rows = []; pw = []; pm = []
    for m in BASE:
        diff = E[m] - E["EVIDENT"]
        w = st.wilcoxon(E["EVIDENT"], E[m], zero_method="wilcox") if np.any(diff != 0) else None
        nz = diff[diff != 0]; rk = st.rankdata(np.abs(nz))
        rbc = float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()) if len(nz) else 0.0
        we = (D["EVIDENT"] >= 0) & (D["EVIDENT"] != ty); wb = (D[m] >= 0) & (D[m] != ty)
        b01 = int(np.sum(~we & wb)); b10 = int(np.sum(we & ~wb))
        mc = st.binomtest(b10, b01 + b10, 0.5).pvalue if (b01 + b10) > 0 else 1.0
        rows.append({"method": m, "energy_med_diff": float(np.median(diff)), "wilcoxon_p": float(w.pvalue) if w else 1.0,
                     "rank_biserial": rbc, "mcnemar_b01": b01, "mcnemar_b10": b10, "mcnemar_p": float(mc),
                     "err_base": float(wb.mean()), "err_evident": float(we.mean())})
        pw.append(rows[-1]["wilcoxon_p"]); pm.append(rows[-1]["mcnemar_p"])
    for r_, a, b in zip(rows, holm(pw), holm(pm)):
        r_["wilcoxon_p_holm"] = float(a); r_["mcnemar_p_holm"] = float(b)
    we = (D["EVIDENT"] >= 0) & (D["EVIDENT"] != ty)
    tests[setname] = {"rows": rows, "n_tracks": int(len(ty)),
                      "evident_errors": int(we.sum()),
                      "cp_upper_all": cp_upper(int(we.sum()), len(we)),
                      "cp_upper_reg": cp_upper(int(we[grp == 0].sum()), int((grp == 0).sum())),
                      "cp_upper_warn": cp_upper(int(we[grp == 1].sum()), int((grp == 1).sum())),
                      "n_reg": int((grp == 0).sum()), "n_warn": int((grp == 1).sum())}
S["tests"] = tests
# energy saving of the controller (paired, per track)
E = pooled("test", "energy"); D = pooled("test", "dec")
S["saving_vs_static"] = float(1 - E["EVIDENT"].sum() / E["EVIDENT-static"].sum())
S["saving_per_commit_vs_static"] = float(1 - (E["EVIDENT"].sum() / (D["EVIDENT"] >= 0).sum()) /
                                         (E["EVIDENT-static"].sum() / (D["EVIDENT-static"] >= 0).sum()))
S["saving_vs_vote"] = float(1 - E["EVIDENT"].sum() / E["Vote+hysteresis"].sum())
S["saving_vs_vote_maha"] = float(1 - E["EVIDENT"].sum() / E["Vote+hysteresis+Maha"].sum())
S["saving_vs_earlyexit"] = float(1 - E["EVIDENT"].sum() / E["EarlyExit+vote"].sum())
# width at commitment (distance proxy)
wdec = np.concatenate([np.load(f"runs/pertrack_s{s}.npz")["test|EVIDENT|wdec"] for s in SEEDS])
S["w_commit_median"] = float(np.median(wdec[D["EVIDENT"] >= 0]))
S["w_commit_iqr"] = [float(np.quantile(wdec[D["EVIDENT"] >= 0], q)) for q in (0.25, 0.75)]
json.dump(S, open("runs/stats.json", "w"), indent=1)
print("stats written; seeds", SEEDS)
print("saving vs static", S["saving_vs_static"], "per commit", S["saving_per_commit_vs_static"], "vs vote", S["saving_vs_vote"],
      "vs early-exit", S["saving_vs_earlyexit"])
for k, v in tests.items():
    print(k, "errors", v["evident_errors"], "/", v["n_tracks"], "CP", v["cp_upper_all"], v["cp_upper_reg"], v["cp_upper_warn"])
    for r_ in v["rows"]:
        print("  %-26s dE=%.3f p=%.2g r=%.2f | errB=%.3f errE=%.3f mcn p=%.2g" % (r_["method"], r_["energy_med_diff"], r_["wilcoxon_p_holm"],
              r_["rank_biserial"], r_["err_base"], r_["err_evident"], r_["mcnemar_p_holm"]))

# ------------------------------------------------ per-operating-point frame accuracy (clean / night)
from common import OPS
acc = {"test": [], "test_night": []}
for s in SEEDS:
    d = np.load(f"runs/outputs_s{s}.npz", allow_pickle=True)
    for nm in acc:
        lp = d[f"{nm}__lp"]; y = d[f"{nm}__y"]
        acc[nm].append([float(np.mean(lp[:, o].argmax(1) == y)) for o in range(len(OPS))])
S["op_acc"] = {nm: {"mean": np.mean(v, 0).tolist(), "sd": np.std(v, 0, ddof=1).tolist() if len(v) > 1 else [0] * len(OPS)} for nm, v in acc.items()}
# reference pool sizes per bin (full op) -> evidence ceiling per frame
d = np.load(f"runs/outputs_s{SEEDS[0]}.npz", allow_pickle=True)
from evident import size_bin
b = size_bin(d["norm__w"]); S["pool_bins"] = [int((b == k).sum()) for k in range(4)]
json.dump(S, open("runs/stats.json", "w"), indent=1)
print("op acc", S["op_acc"]["test"]["mean"], "pool", S["pool_bins"])
