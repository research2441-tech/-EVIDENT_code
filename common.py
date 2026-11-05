"""Shared components for EVIDENT: multi-exit backbone, operating points, data splits,
augmentation, test-time corruptions and synthetic approach tracks."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HELD_OUT = [0, 19, 21, 24, 27, 29, 32, 41]          # vocabulary gaps (open-world classes)
KNOWN = [c for c in range(43) if c not in HELD_OUT]  # 35 known classes
K2I = {c: i for i, c in enumerate(KNOWN)}
# Safety groups used for group-conditional risk levels
REGULATORY = {1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 17, 33, 34, 35, 36, 37, 38, 39, 40}
def group_of(gtsrb_id):
    return 0 if gtsrb_id in REGULATORY else 1     # 0 = regulatory, 1 = warning / other

RES = (32, 48)          # input resolutions
EXITS = (1, 2, 3)       # exit depths
OPS = [(k, r) for r in RES for k in EXITS]   # 6 operating points, index o


# ----------------------------------------------------------------------------- model
class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, 1, 1, bias=False); self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(c)
    def forward(self, x):
        return F.relu(x + self.b2(self.c2(F.relu(self.b1(self.c1(x))))))

def down(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, 2, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

class Head(nn.Module):
    def __init__(self, ci, hid, n):
        super().__init__()
        self.p = nn.Sequential(nn.Conv2d(ci, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.ReLU(inplace=True))
        self.fc = nn.Linear(hid, n)
    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(self.p(x), 1).flatten(1))

class MultiExitNet(nn.Module):
    """Three-exit residual CNN. Exit k reuses all computation of exits < k (cascade)."""
    def __init__(self, n=len(KNOWN)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 24, 3, 1, 1, bias=False), nn.BatchNorm2d(24), nn.ReLU(inplace=True))
        self.s1 = nn.Sequential(down(24, 32), Res(32))
        self.s2 = nn.Sequential(down(32, 64), Res(64))
        self.s3 = nn.Sequential(down(64, 128), Res(128))
        self.h1, self.h2, self.h3 = Head(32, 64, n), Head(64, 96, n), Head(128, 160, n)
        self.drop = nn.Dropout(0.1)
    def forward(self, x, upto=3):
        out = []
        f = self.s1(self.stem(x)); out.append(self.h1(f))
        if upto >= 2:
            f = self.s2(f); out.append(self.h2(f))
        if upto >= 3:
            f = self.s3(f); out.append(self.h3(self.drop(f)))
        return out

MEAN = torch.tensor([0.34, 0.31, 0.32]).view(1, 3, 1, 1)
STD = torch.tensor([0.27, 0.26, 0.27]).view(1, 3, 1, 1)

def to_tensor(xu8, res):
    x = torch.from_numpy(xu8).permute(0, 3, 1, 2).float() / 255.0
    if res != x.shape[-1]:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True)
    return (x - MEAN) / STD


# ----------------------------------------------------------------------------- splits
def track_splits(tid, y, seed, frac=(0.55, 0.20, 0.25)):
    """Track-disjoint, class-stratified split of known-class tracks into
    train / track-calibration (B) / test. Held-out-class tracks form the open-world set."""
    rng = np.random.default_rng(seed)
    u, first = np.unique(tid, return_index=True)
    uy = y[first]
    tr, ca, te = [], [], []
    for c in KNOWN:
        t = rng.permutation(u[uy == c]); n = len(t)
        a = max(1, int(round(frac[0] * n))); b = max(1, int(round(frac[1] * n)))
        tr += list(t[:a]); ca += list(t[a:a + b]); te += list(t[a + b:])
    unk = list(u[np.isin(uy, HELD_OUT)])
    return np.array(tr), np.array(ca), np.array(te), np.array(unk)


# ----------------------------------------------------------------------------- augmentation (training only)
def augment(xb, rng):
    """xb: float tensor B x 3 x 48 x 48 in [0,1]. Simulates distance (down-up sampling),
    viewpoint jitter, illumination and focus changes."""
    B = xb.shape[0]
    out = []
    for i in range(B):
        x = xb[i:i + 1]
        s = int(rng.integers(14, 49))
        if s < 48:
            x = F.interpolate(F.interpolate(x, size=(s, s), mode="bilinear", antialias=True), size=(48, 48), mode="bilinear")
        out.append(x)
    x = torch.cat(out)
    ang = torch.from_numpy(rng.uniform(-0.17, 0.17, B)).float()
    sc = torch.from_numpy(rng.uniform(0.9, 1.1, B)).float()
    tx = torch.from_numpy(rng.uniform(-0.08, 0.08, (B, 2))).float()
    th = torch.zeros(B, 2, 3)
    th[:, 0, 0] = torch.cos(ang) / sc; th[:, 0, 1] = -torch.sin(ang) / sc
    th[:, 1, 0] = torch.sin(ang) / sc; th[:, 1, 1] = torch.cos(ang) / sc
    th[:, :, 2] = tx
    g = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, g, padding_mode="border", align_corners=False)
    br = torch.from_numpy(rng.uniform(0.6, 1.4, (B, 1, 1, 1))).float()
    ct = torch.from_numpy(rng.uniform(0.7, 1.3, (B, 1, 1, 1))).float()
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - m) * ct + m) * br
    return x.clamp(0, 1)


# ----------------------------------------------------------------------------- test-time corruptions (severity 1..5)
def _blur_kernel_motion(L):
    k = np.zeros((L, L), np.float32); k[L // 2, :] = 1.0 / L
    return k

def corrupt(x, kind, sev, rng):
    """x: uint8 N x 48 x 48 x 3. Returns corrupted uint8 array. Deterministic given rng."""
    f = x.astype(np.float32) / 255.0
    if kind == "night":                       # low light: gamma darkening + shot/read noise
        g = [1.6, 2.0, 2.5, 3.0, 3.6][sev - 1]; s = [0.02, 0.03, 0.045, 0.06, 0.08][sev - 1]
        f = f ** g
        f = f + rng.normal(0, s, f.shape) + rng.poisson(f * 60) / 60.0 - f
    elif kind == "haze":                      # atmospheric veil
        a = [0.3, 0.4, 0.5, 0.6, 0.7][sev - 1]
        f = f * (1 - a) + a * 0.8
    elif kind == "motion":                    # horizontal motion blur (box kernel of length L)
        L = [3, 5, 7, 9, 11][sev - 1]; p = L // 2
        fp = np.pad(f, ((0, 0), (0, 0), (p, p), (0, 0)), mode="edge")
        c = np.cumsum(fp, axis=2); c = np.concatenate([np.zeros_like(c[:, :, :1]), c], 2)
        f = (c[:, :, L:] - c[:, :, :-L]) / L
    elif kind == "jpeg":
        from PIL import Image
        import io
        q = [25, 18, 12, 8, 5][sev - 1]
        out = []
        for n in range(len(x)):
            b = io.BytesIO(); Image.fromarray(x[n]).save(b, "JPEG", quality=q)
            out.append(np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB")))
        return np.stack(out)
    else:
        raise ValueError(kind)
    return (np.clip(f, 0, 1) * 255).round().astype(np.uint8)


# ----------------------------------------------------------------------------- synthetic approach tracks
def synth_track(img, rng, T=30, w0=18.0, w1=None):
    """Turn one 48x48 sign crop into an approach sequence of T frames whose effective
    resolution grows geometrically from w0 px (far) to w1 px (near)."""
    from PIL import Image
    w1 = w1 or float(rng.uniform(60, 110))
    ws = w0 * (w1 / w0) ** (np.arange(T) / (T - 1))
    im = Image.fromarray(img)
    frames = []
    for w in ws:
        s = int(max(8, min(48, round(w * 48 / 64))))     # sign occupies ~75% of the crop
        dx, dy = rng.integers(-1, 2, 2)
        f = im.resize((s, s), Image.BILINEAR).resize((48, 48), Image.BILINEAR)
        a = np.roll(np.asarray(f).astype(np.float32), (dy, dx), (0, 1))
        a = a + rng.normal(0, 2.0, a.shape)
        frames.append(np.clip(a, 0, 255).astype(np.uint8))
    return np.stack(frames), ws.astype(np.float32)
