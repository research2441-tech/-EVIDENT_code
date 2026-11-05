"""Tiny grayscale diagram engine that refuses overlapping boxes, text that does not fit its
box, and arrows that pass through boxes other than their own endpoints."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.path import Path

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif"})


class Diagram:
    def __init__(self, w, h, figw):
        self.W, self.H = w, h
        self.fig = plt.figure(figsize=(figw, figw * h / w))
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis("off")
        self.boxes = {}; self.arrows = []; self.texts = []

    def box(self, name, x, y, w, h, text, fill="#FFFFFF", lw=1.0, fs=7.5, bold=False, dashed=False, round_=True):
        for n, (bx, by, bw, bh) in self.boxes.items():
            if x < bx + bw + 0.05 and bx < x + w + 0.05 and y < by + bh + 0.05 and by < y + h + 0.05:
                raise ValueError(f"box {name} overlaps {n}")
        style = "round,pad=0.02,rounding_size=0.08" if round_ else "square,pad=0.0"
        p = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fill, ec="black", lw=lw,
                           ls=(0, (4, 2)) if dashed else "-")
        self.ax.add_patch(p)
        t = self.ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                         fontweight="bold" if bold else "normal", linespacing=1.25)
        self.boxes[name] = (x, y, w, h); self.texts.append((name, t))

    def frame(self, x, y, w, h, label, fs=7):
        r = Rectangle((x, y), w, h, fc="none", ec="#555555", lw=0.8, ls=(0, (5, 3)))
        self.ax.add_patch(r)
        self.ax.text(x + 0.08, y + h - 0.08, label, ha="left", va="top", fontsize=fs, style="italic", color="#333333")

    def anchor(self, name, side, frac=0.5):
        x, y, w, h = self.boxes[name]
        return {"l": (x, y + h * frac), "r": (x + w, y + h * frac), "t": (x + w * frac, y + h),
                "b": (x + w * frac, y)}[side]

    def arrow(self, pts, src=None, dst=None, dashed=False, label=None, lpos=None, fs=6.5, lw=1.0, head=True):
        """pts: list of (x,y); straight segments; arrow head at the end."""
        self.arrows.append((pts, src, dst))
        ls = (0, (4, 2.5)) if dashed else "-"
        for i in range(len(pts) - 1):
            last = i == len(pts) - 2
            a = FancyArrowPatch(pts[i], pts[i + 1], arrowstyle="-|>" if (last and head) else "-",
                                mutation_scale=9, lw=lw, color="black", ls=ls, shrinkA=0, shrinkB=0)
            self.ax.add_patch(a)
        if label:
            lx, ly = lpos
            self.ax.text(lx, ly, label, fontsize=fs, ha="center", va="center",
                         bbox=dict(fc="white", ec="none", pad=0.6))

    def note(self, x, y, s, fs=6.5, ha="center", style="normal"):
        self.ax.text(x, y, s, fontsize=fs, ha=ha, va="center", style=style)

    def check(self):
        self.fig.canvas.draw(); r = self.fig.canvas.get_renderer()
        inv = self.ax.transData.inverted()
        for name, t in self.texts:
            bb = t.get_window_extent(r); (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            x, y, w, h = self.boxes[name]
            if x0 < x + 0.03 or x1 > x + w - 0.03 or y0 < y + 0.02 or y1 > y + h - 0.02:
                raise ValueError(f"text does not fit box {name}: {(x0, x1, y0, y1)} vs {(x, x + w, y, y + h)}")
        for pts, src, dst in self.arrows:
            for i in range(len(pts) - 1):
                (ax_, ay), (bx_, by_) = pts[i], pts[i + 1]
                for n, (x, y, w, h) in self.boxes.items():
                    if n in (src, dst):
                        continue
                    for s in [j / 40 for j in range(1, 40)]:
                        px, py = ax_ + s * (bx_ - ax_), ay + s * (by_ - ay)
                        if x + 0.02 < px < x + w - 0.02 and y + 0.02 < py < y + h - 0.02:
                            raise ValueError(f"arrow {src}->{dst} crosses box {n}")

    def save(self, path):
        self.check()
        self.fig.savefig(path, dpi=600)
        plt.close(self.fig)
