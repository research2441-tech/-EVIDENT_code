"""Result figures (grayscale) from runs/results.json and per-track records."""
import json, sys, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
sys.path.insert(0, ".")

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                     "font.size": 7.5, "axes.linewidth": 0.7, "axes.labelsize": 7.5, "xtick.labelsize": 6.8,
                     "ytick.labelsize": 6.8, "legend.fontsize": 6.4, "lines.linewidth": 1.1,
                     "axes.spines.top": False, "axes.spines.right": False})
RUNS = "runs"; OUT = "out"
R = json.load(open(f"{RUNS}/results.json"))
SEEDS = [r["seed"] for r in R]
MK = {"EVIDENT": ("o", "black", "-"), "EVIDENT-static": ("s", "black", "--"), "EVIDENT-label-only": ("D", "#555555", ":"),
      "Single-frame@deadline": ("v", "#777777", "-"), "Vote+hysteresis": ("^", "#444444", "-"),
      "Vote+hysteresis+MSP": ("<", "#666666", "-"), "Vote+hysteresis+Maha": (">", "#333333", "-"),
      "EarlyExit+vote": ("P", "#888888", "-"), "Frame-CP first-singleton": ("X", "#555555", "-"),
      "E-process (Ville)": ("*", "#222222", "-"), "Bayes-SPRT (tuned)": ("h", "#999999", "-")}
SHORT = {"EVIDENT": "EVIDENT", "EVIDENT-static": "EVIDENT-static", "EVIDENT-label-only": "Label-only",
         "Single-frame@deadline": "Single frame", "Vote+hysteresis": "Vote+hyst.", "Vote+hysteresis+MSP": "Vote+MSP",
         "Vote+hysteresis+Maha": "Vote+Maha", "EarlyExit+vote": "Early-exit+vote", "Frame-CP first-singleton": "Frame-CP",
         "E-process (Ville)": "Ville e-proc.", "Bayes-SPRT (tuned)": "Bayes-SPRT"}


def agg(path):
    """mean and 95% t-interval across seeds of R[...][path]"""
    vals = []
    for r in R:
        v = r
        for p in path:
            v = v[p]
        vals.append(v)
    v = np.array(vals, float); m = v.mean()
    h = 2.776 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return m, h


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=600, bbox_inches="tight", pad_inches=0.03); plt.close(fig)


def fig_tradeoff():
    """energy per track vs committed error, clean and night (two stacked panels)."""
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.6))
    for ax, s, ttl in zip(axes, ("test", "test_night"), ("(a) Clean test tracks", "(b) Night corruption (severity 3)")):
        for m, (mk, col, _) in MK.items():
            ex, eh = agg(("sets", s, m, "energy_mJ")); yy, yh = agg(("sets", s, m, "committed_error"))
            ax.errorbar(ex, 100 * yy, yerr=100 * yh, fmt=mk, color=col, ms=5 if m != "EVIDENT" else 6.5,
                        mfc="white" if m not in ("EVIDENT", "EVIDENT-static") else col, mew=0.9, elinewidth=0.6,
                        capsize=1.5, label=SHORT[m], zorder=3 if m == "EVIDENT" else 2)
        ax.set_xlabel("Energy per track (mJ)"); ax.set_ylabel("Committed error (%)")
        ax.set_title(ttl, fontsize=7.5, loc="left"); ax.grid(alpha=0.25, lw=0.4)
        ax.set_yscale("symlog", linthresh=1.0); ax.set_ylim(-0.2, 100)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), frameon=False, columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout(h_pad=1.6)
    save(fig, "fig3_tradeoff.png")


def fig_validity():
    """(a) realised committed error vs nominal alpha; (b) re-split envelope violation histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    ax = axes[0]; al = [0.01, 0.02, 0.05, 0.1, 0.2]; xx = np.arange(len(al))
    for j, (nm, key, mk, ls) in enumerate((("EVIDENT", "EVIDENT", "o", "-"), ("Label-only", "EVIDENT-label-only", "D", ":"),
                            ("Frame-CP", "Frame-CP", "X", "-"), ("Ville e-proc.", "Ville", "*", "--"))):
        m = []; h = []; xs = []
        for i, a in enumerate(al):
            if key.startswith("EVIDENT") and a == 0.01:
                continue          # infeasible: fewer than 1/alpha - 1 calibration tracks in the warning group
            mm, hh = agg(("sensitivity", f"alpha={a}", key, "committed_error")); m.append(100 * mm); h.append(100 * hh); xs.append(i)
        ax.errorbar(np.array(xs) + (j - 1.5) * 0.06, m, yerr=h, marker=mk, ls=ls, color="black", ms=4.5, mfc="white",
                    capsize=1.5, elinewidth=0.6, label=nm)
    ax.plot(xx, [100 * a for a in al], color="#999999", lw=0.9, ls="--", label="nominal $\\alpha$")
    ax.set_xticks(xx); ax.set_xticklabels([f"{100 * a:g}" for a in al]); ax.set_ylim(-0.3, 6.0)
    ax.set_xlabel("Nominal level $\\alpha$ (%), both groups"); ax.set_ylabel("Realised committed error (%)")
    ax.set_title("(a) Error of the stopped decision", fontsize=7.5, loc="left"); ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax = axes[1]
    vg = np.concatenate([np.array(r["resplit"]["viol_g"]) for r in R])
    fl = np.concatenate([np.array(r["resplit"]["flag"]) for r in R])
    bins = np.linspace(0, 0.14, 36)
    ax.hist(100 * vg[:, 0], bins=100 * bins, histtype="stepfilled", color="#BBBBBB", ec="black", lw=0.6, label="regulatory, $\\alpha_g$=2%")
    ax.hist(100 * vg[:, 1], bins=100 * bins, histtype="step", color="black", lw=1.0, label="warning/other, $\\alpha_g$=5%")
    ax.hist(100 * fl, bins=100 * bins, histtype="step", color="#555555", lw=1.0, ls="--", label="typicality flag, $\\alpha_\\tau$=5%")
    for a, ls in ((2, "-"), (5, "--")):
        ax.axvline(a, color="#444444", lw=0.8, ls=ls)
    ax.set_xlabel("Envelope violation on test split (%)"); ax.set_ylabel("Re-partitions")
    ax.set_title(f"(b) {len(vg)} random calibration/test re-partitions", fontsize=7.5, loc="left")
    ax.legend(frameon=False, loc="upper right"); ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig4_validity.png")


def fig_robust():
    """grouped bars: correct commitment, committed error and no commitment across shifts."""
    sets = ["test", "test_haze", "test_motion", "test_jpeg", "test_night", "indian"]
    lab = ["Clean", "Haze", "Motion\nblur", "JPEG", "Night", "Indian\n(mapped)"]
    meths = ["EVIDENT", "EVIDENT-label-only", "Vote+hysteresis", "Vote+hysteresis+Maha", "E-process (Ville)"]
    hatch = ["", "////", "....", "xxxx", "\\\\\\\\"]; fill = ["black", "white", "#DDDDDD", "white", "#999999"]
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 4.9), sharex=True)
    x = np.arange(len(sets)); wd = 0.16
    for j, m in enumerate(meths):
        c = [agg(("sets", s, m, "commit_correct")) for s in sets]
        e = [agg(("sets", s, m, "committed_error")) for s in sets]
        nc_ = []
        for s in sets:
            v = np.array([r["sets"][s][m]["abstain"] + r["sets"][s][m]["defer"] for r in R])
            nc_.append((v.mean(), 2.776 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0))
        for ax, vals in zip(axes, (c, e, nc_)):
            ax.bar(x + (j - 2) * wd, [100 * v[0] for v in vals], wd, yerr=[100 * v[1] for v in vals], color=fill[j], ec="black",
                   lw=0.6, hatch=hatch[j], error_kw=dict(lw=0.6, capsize=1.2), label=SHORT[m])
    axes[0].set_ylabel("Correct\ncommitment (%)"); axes[1].set_ylabel("Committed\nerror (%)"); axes[2].set_ylabel("No commitment\n(abstain+defer, %)")
    axes[2].set_xticks(x); axes[2].set_xticklabels(lab)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25, lw=0.4)
    axes[0].legend(ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.03), columnspacing=1.0)
    fig.tight_layout(h_pad=0.6)
    save(fig, "fig6_robustness.png")


def fig_openworld():
    """known-sign commitment vs wrong commitment on unseen classes; a_conf sweep + baselines."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax = axes[0]
    acs = [0.0, 1.0, 2.0, 4.0, 6.0]
    xs = [100 * agg(("sensitivity", f"a_conf={a}", "test", "commit_correct"))[0] for a in acs]
    ys = [100 * agg(("sensitivity", f"a_conf={a}", "unk", "unk_wrong_commit"))[0] for a in acs]
    ax.plot(xs, ys, "-o", color="black", ms=4.5, label="EVIDENT, confirmation $a$")
    ax.annotate(f"$a$={acs[0]:g}", (xs[0], ys[0]), textcoords="offset points", xytext=(4, 6), fontsize=6)
    ax.annotate(f"$a$={acs[-1]:g}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(-4, 7), fontsize=6)
    for m in ("EVIDENT-static", "EVIDENT-label-only", "Vote+hysteresis", "Vote+hysteresis+MSP", "Vote+hysteresis+Maha",
              "Frame-CP first-singleton", "E-process (Ville)"):
        mk, col, _ = MK[m]
        ax.plot(100 * agg(("sets", "test", m, "commit_correct"))[0], 100 * agg(("sets", "unk", m, "unk_wrong_commit"))[0],
                mk, color=col, mfc="white", ms=5, ls="none", label=SHORT[m])
    ax.set_xlabel("Correct commitment, known test signs (%)"); ax.set_ylabel("Wrong commitment, held-out classes (%)")
    ax.set_title("(a) GTSRB held-out classes", fontsize=7.5, loc="left"); ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, fontsize=5.8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax = axes[1]
    meths = ["EVIDENT", "EVIDENT-static", "Vote+hysteresis+MSP", "Vote+hysteresis+Maha", "Frame-CP first-singleton", "Vote+hysteresis"]
    ab = [100 * agg(("sets", "indian", m, "unk_abstain"))[0] for m in meths]
    wr = [100 * agg(("sets", "indian", m, "unk_wrong_commit"))[0] for m in meths]
    y = np.arange(len(meths))
    ax.barh(y, ab, color="#BBBBBB", ec="black", lw=0.6, label="abstain")
    ax.barh(y, wr, left=ab, color="white", ec="black", lw=0.6, hatch="////", label="wrong commit")
    ax.set_yticks(y); ax.set_yticklabels([SHORT[m] for m in meths]); ax.invert_yaxis()
    ax.set_xlabel("Indian signs without a GTSRB class (%)"); ax.set_xlim(0, 100)
    ax.set_title("(b) Cross-region unknown signs", fontsize=7.5, loc="left", pad=14)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=2, fontsize=6)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig7_openworld.png")


def fig_controller():
    """Lagrangian frontier on calibration tracks (seed 0) and test lead-time distribution."""
    r = R[0]; g = np.array([[e[1], e[2]] for e in r["grid"]]); th = r["theta"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    ax = axes[0]
    ax.scatter(g[:, 0], 100 * g[:, 1], s=9, facecolor="white", edgecolor="#555555", lw=0.6, label="configurations $\\theta\\in\\Theta$")
    pts = g[np.argsort(g[:, 0])]; hull = []; best = 1e9
    for e, dd in pts:
        if dd < best - 1e-12:
            hull.append((e, dd)); best = dd
    hull = np.array(hull)
    ax.plot(hull[:, 0], 100 * hull[:, 1], "-", color="black", lw=0.9, label="Pareto frontier")
    sel = [e for e in r["grid"] if e[0] == th][0]
    ax.plot(sel[1], 100 * sel[2], "o", color="black", ms=7, label="$\\theta^\\star$ (dual bisection)")
    ax.axhline(100 * r["beta"], color="#777777", ls="--", lw=0.8, label="budget $\\bar{D}(\\theta_0)+\\delta$")
    ax.set_xlabel("Mean energy per calibration track (mJ)"); ax.set_ylabel("Non-commitment, known (%)")
    ax.set_title("(a) Controller search, seed 0", fontsize=7.5, loc="left"); ax.grid(alpha=0.25, lw=0.4)
    ax.set_ylim(0, 100 * g[:, 1].max() * 1.45)
    ax.legend(frameon=False, loc="upper right", ncol=2, columnspacing=0.8)
    ax = axes[1]
    for m, ls in (("EVIDENT", "-"), ("EVIDENT-static", "--"), ("Vote+hysteresis", ":"), ("Single-frame@deadline", "-.")):
        vals = []
        for s in SEEDS:
            p = np.load(f"{RUNS}/pertrack_s{s}.npz"); d = p[f"test|{m}|dec"]; dl = p["test|dl"]; t = p[f"test|{m}|t"]
            vals.append((dl - 1 - t)[d >= 0])
        v = np.sort(np.concatenate(vals))
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color="black", ls=ls, lw=1.0, label=SHORT[m])
    ax.set_xlabel("Frames between commitment and deadline"); ax.set_ylabel("Cumulative fraction of commitments")
    ax.set_title("(b) Decision lead, clean test tracks", fontsize=7.5, loc="left"); ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig5_controller.png")


def fig_trajectory():
    """evidence trajectories for two real test tracks (seed 0): a known regulatory sign and a night frame track."""
    from evident import Normaliser, TypNormaliser, Tracks, log_evidence, log_e, deadlines, ADMISSIBLE
    from common import KNOWN
    d = np.load(f"{RUNS}/outputs_s0.npz", allow_pickle=True); typ = np.load(f"{RUNS}/typ_s0.npz")
    norm = Normaliser(d["norm__lp"], d["norm__y"], d["norm__w"]); tn = TypNormaliser(typ["norm"], d["norm__w"])
    r0 = R[0]; lam = r0["lam"]; lam_t = r0["lam_t"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.6), sharex="col")
    for col, (name, pick) in enumerate((("test", "known"), ("unk", "unknown"))):
        tr = Tracks(d, name, typ); p = norm.pvalues(tr.lp, tr.w); ev = log_evidence(p, 0.02)
        lt = log_e(tn.pvalues(tr.dist, tr.w), 0.1); dl = deadlines(tr)
        # choose a representative track: longest usable horizon among speed-limit (known) / any (unknown)
        cand = [i for i in range(tr.n) if dl[i] >= 18 and (tr.tgid[i] in (1, 2, 3, 4, 5) if pick == "known" else True)]
        if pick == "known":
            i = cand[0]
        else:
            flagged = [j for j in cand if np.cumsum(np.maximum(lt[tr.frames(j)][: dl[j], 5], 0))[-1] > lam_t]
            i = flagged[0] if flagged else cand[0]
        f = tr.frames(i); o = 5
        A = np.cumsum(ev[f][: dl[i], o, :], 0); S = np.cumsum(lt[f][: dl[i], o]); Pp = np.cumsum(np.maximum(lt[f][: dl[i], o], 0))
        t = np.arange(1, dl[i] + 1)
        ax = axes[0, col]
        order = np.argsort(A[-1])
        for j in order[:6]:
            lw = 1.6 if (tr.ty[i] >= 0 and j == tr.ty[i]) else 0.8
            ax.plot(t, A[:, j], color="black" if lw > 1 else "#888888", lw=lw)
        ax.axhline(lam[0], color="black", ls="--", lw=0.8); ax.axhline(lam[1], color="#666666", ls="-.", lw=0.8)
        ax.set_ylim(-0.3, max(12, max(lam) * 1.9))
        ax.set_ylabel("Label evidence $A_t(y)$")
        ttl = f"(a) Known sign, GTSRB class {tr.tgid[i]}" if pick == "known" else f"(b) Unseen sign, held-out GTSRB class {tr.tgid[i]}"
        ax.set_title(ttl, fontsize=7.5, loc="left"); ax.grid(alpha=0.25, lw=0.4)
        tr_ = ax.get_yaxis_transform()
        ax.text(1.01, lam[0], "$\\lambda_{\\mathrm{reg}}$", transform=tr_, va="center", fontsize=6.5, clip_on=False)
        ax.text(1.01, lam[1], "$\\lambda_{\\mathrm{warn}}$", transform=tr_, va="center", fontsize=6.5, color="#444444", clip_on=False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax = axes[1, col]
        ax.plot(t, S, color="black", lw=1.1, label="signed $S_t$"); ax.plot(t, Pp, color="#666666", lw=1.1, ls=":", label="rectified $P_t$")
        ax.axhline(lam_t, color="black", ls="--", lw=0.8); ax.axhline(-2, color="#777777", ls="-.", lw=0.8)
        tr_ = ax.get_yaxis_transform()
        ax.text(1.01, lam_t, "$\\lambda_\\tau$", transform=tr_, va="center", fontsize=6.5, clip_on=False)
        ax.text(1.01, -2, "$-a$", transform=tr_, va="center", fontsize=6.5, clip_on=False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel("Frame index $t$ (full operating point)"); ax.set_ylabel("Typicality evidence")
        ax.grid(alpha=0.25, lw=0.4)
        if col == 0:
            ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(h_pad=0.6, w_pad=2.0)
    save(fig, "fig8_trajectories.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_tradeoff(); fig_validity(); fig_controller(); fig_robust(); fig_openworld(); fig_trajectory()
    print("ok")
