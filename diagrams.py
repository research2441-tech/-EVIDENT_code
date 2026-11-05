"""Fig. 1 (EVIDENT inference + offline calibration) and Fig. 2 (multi-exit training)."""
from layout import Diagram

GREY = "#EDEDED"; LIGHT = "#F7F7F7"


def fig_system(path):
    D = Diagram(14, 8.3, 7.0)
    # row 0: offline controller tuning
    D.box("theta", 0.7, 7.05, 4.75, 0.95, "Controller tuning on calibration tracks\n(dual bisection) $\\theta^\\star$", fill=LIGHT, dashed=True, fs=7)
    D.box("cost", 6.35, 7.05, 3.7, 0.95, "Device cost profile $c_o$\n(MAC, DRAM, SRAM energy)", fill=LIGHT, dashed=True, fs=7)
    # row 1: online per-frame chain
    D.box("x", 0.15, 5.35, 1.95, 1.1, "Tracked sign\ncrop $x_t$, width $w_t$", fs=7)
    D.box("ctrl", 2.55, 5.35, 2.0, 1.1, "Controller $\\pi_\\theta$\nskip / depth $k$", fill=GREY, bold=False, fs=7)
    D.box("net", 5.0, 5.35, 2.25, 1.1, "Multi-exit backbone\nexit $k$, input $r(w_t)$", fs=7)
    D.box("pv", 7.7, 5.35, 2.35, 1.1, "Size-conditional\nconformal $p$-values", fill=GREY, fs=7)
    D.box("ref", 10.65, 5.35, 3.2, 1.1, "Reference scores $\\mathcal{N}_{o,b}$\n(normalisation frames)", fill=LIGHT, dashed=True, fs=7)
    # row 2
    D.box("ell", 7.7, 3.7, 2.35, 1.1, "Rectified log e-value\n$\\ell_t(y)=(\\log e_t(y))^{+}$", fill=GREY, fs=7)
    D.box("acc", 10.65, 3.7, 3.2, 1.1, "Evidence process\n$A_t(y)=A_{t-1}(y)+\\ell_t(y)$", fill=GREY, fs=7)
    # row 3
    D.box("dec", 4.6, 2.0, 5.45, 1.1, "Decision: $|C_t|=1$ and $S_t\\leq -a\\Rightarrow$ commit $\\hat{y}$;\n$C_t=\\emptyset$ or $P_t>\\lambda_\\tau\\Rightarrow$ abstain (unknown or atypical);\ndeadline reached $\\Rightarrow$ defer and report $C_t$", fs=6.6)
    D.box("set", 10.65, 2.0, 3.2, 1.1, "Prediction set\n$C_t=\\{y: A_t(y)\\leq\\lambda_g\\}$", fill=GREY, fs=7)
    # row 4: offline calibration
    D.box("cal", 3.4, 0.25, 3.05, 1.05, "Calibration tracks\n(known signs, disjoint)", fill=LIGHT, dashed=True, fs=7)
    D.box("env", 6.95, 0.25, 3.1, 1.05, "Envelope score\n$R_i=\\Sigma_t\\,\\mathrm{max}_o\\,\\ell_{t,o}(y_i)$", fill=LIGHT, dashed=True, fs=7)
    D.box("lam", 10.65, 0.25, 3.2, 1.05, "Group quantile $\\lambda_g$\n(level $\\alpha_g$)", fill=LIGHT, dashed=True, fs=7)
    # arrows (solid = data flow, dashed = per-frame feedback loop)
    D.arrow([D.anchor("cost", "l"), D.anchor("theta", "r")], "cost", "theta", label="$c_o$", lpos=(5.9, 7.78))
    D.arrow([(3.55, 7.05), (3.55, 6.45)], "theta", "ctrl", label="$\\theta^\\star$", lpos=(3.95, 6.75))
    D.arrow([D.anchor("x", "r"), D.anchor("ctrl", "l")], "x", "ctrl")
    D.arrow([D.anchor("ctrl", "r"), D.anchor("net", "l")], "ctrl", "net", label="$o_t$", lpos=(4.78, 6.7))
    D.arrow([D.anchor("net", "r"), D.anchor("pv", "l")], "net", "pv", label="$\\pi_o$", lpos=(7.48, 6.7))
    D.arrow([D.anchor("ref", "l"), D.anchor("pv", "r")], "ref", "pv")
    D.arrow([D.anchor("pv", "b"), D.anchor("ell", "t")], "pv", "ell", label="$p_t(y)$", lpos=(9.35, 5.03))
    D.arrow([D.anchor("ell", "r"), D.anchor("acc", "l")], "ell", "acc")
    D.arrow([D.anchor("acc", "b"), D.anchor("set", "t")], "acc", "set")
    D.arrow([D.anchor("set", "l"), D.anchor("dec", "r")], "set", "dec")
    D.arrow([D.anchor("lam", "t"), D.anchor("set", "b")], "lam", "set", label="$\\lambda_g$", lpos=(12.7, 1.65))
    D.arrow([D.anchor("cal", "r"), D.anchor("env", "l")], "cal", "env")
    D.arrow([D.anchor("env", "r"), D.anchor("lam", "l")], "env", "lam")
    D.arrow([(4.6, 2.55), (3.55, 2.55), (3.55, 5.35)], "dec", "ctrl", dashed=True,
            label="otherwise:\nframe $t{+}1$", lpos=(2.6, 3.6))
    D.note(0.2, 0.75, "Offline (once)", fs=6.8, ha="left", style="italic")
    D.note(0.2, 4.45, "Online (per frame)", fs=6.8, ha="left", style="italic")
    D.save(path)


def fig_training(path):
    D = Diagram(14, 4.4, 7.0)
    D.box("in", 0.15, 2.25, 2.3, 1.05, "Training frames\n$r\\in\\{32,48\\}$, distance /\npose / light aug.", fs=6.6)
    D.box("s1", 3.0, 2.25, 2.4, 1.05, "Stem + stage 1\n($r/2$, 32 ch)", fill=GREY, fs=7)
    D.box("s2", 6.05, 2.25, 2.4, 1.05, "Stage 2\n($r/4$, 64 ch)", fill=GREY, fs=7)
    D.box("s3", 9.1, 2.25, 2.4, 1.05, "Stage 3\n($r/8$, 128 ch)", fill=GREY, fs=7)
    D.box("h1", 3.0, 0.6, 2.4, 0.95, "Exit 1: $\\pi_{(1,r)}$\n$\\mathrm{CE}_1$", fs=7)
    D.box("h2", 6.05, 0.6, 2.4, 0.95, "Exit 2: $\\pi_{(2,r)}$\n$\\mathrm{CE}_2$", fs=7)
    D.box("h3", 9.1, 0.6, 2.4, 0.95, "Exit 3: $\\pi_{(3,r)}$\n$\\mathrm{CE}_3$", fs=7)
    D.box("L", 12.0, 0.6, 1.85, 0.95, "$\\mathcal{L}=\\Sigma_k\\,\\omega_k \\mathrm{CE}_k$", fs=7)
    D.arrow([D.anchor("in", "r"), D.anchor("s1", "l")], "in", "s1")
    D.arrow([D.anchor("s1", "r"), D.anchor("s2", "l")], "s1", "s2")
    D.arrow([D.anchor("s2", "r"), D.anchor("s3", "l")], "s2", "s3")
    for s, h in (("s1", "h1"), ("s2", "h2"), ("s3", "h3")):
        D.arrow([D.anchor(s, "b"), D.anchor(h, "t")], s, h)
    D.arrow([D.anchor("h3", "r"), D.anchor("L", "l")], "h3", "L")
    D.arrow([D.anchor("h2", "b", 0.5), (7.25, 0.3), (12.6, 0.3), (12.6, 0.6)], "h2", "L", head=True)
    D.arrow([D.anchor("h1", "b", 0.5), (4.2, 0.3), (7.25, 0.3)], "h1", None, head=False)
    # reverse-mode gradient: from the loss back to every stage (dashed)
    D.arrow([D.anchor("L", "t", 0.5), (12.93, 3.85), (4.2, 3.85), (4.2, 3.3)], "L", "s1", dashed=True,
            label=None)
    D.note(8.6, 4.17, "reverse-mode gradient $\\nabla_\\phi\\mathcal{L}$ (dashed)", fs=6.8)
    D.arrow([(7.25, 3.85), (7.25, 3.3)], None, "s2", dashed=True)
    D.arrow([(10.3, 3.85), (10.3, 3.3)], None, "s3", dashed=True)
    D.save(path)


if __name__ == "__main__":
    import os
    os.makedirs("out", exist_ok=True)
    fig_system("out/fig1_system.png")
    fig_training("out/fig2_training.png")
    print("ok")
