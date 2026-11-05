"""Per-operating-point cost profile of the multi-exit backbone.

MACs and parameter counts are counted exactly with forward hooks. Energy follows the
45-nm per-operation figures of Horowitz (ISSCC 2014): 3.7 pJ fp32 multiply + 0.9 pJ fp32 add
per MAC, 1.3 nJ per 64-bit DRAM read (weights streamed once per frame), and 10 pJ per 64-bit
8-kB SRAM access for activations (one write + one read). Latency is measured on this CPU,
batch size 1, two threads, median of repeated runs.
"""
import json, time, numpy as np, torch, torch.nn as nn
from common import MultiExitNet, OPS

E_MAC = 4.6e-12            # J per fp32 multiply-accumulate
E_DRAM_WORD = 0.65e-9      # J per 32-bit word from DRAM
E_SRAM_WORD = 5.0e-12      # J per 32-bit word to/from small SRAM


def count(module, x):
    macs, acts = [0], [0]
    def hook(m, i, o):
        if isinstance(m, nn.Conv2d):
            macs[0] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            acts[0] += o.numel()
        elif isinstance(m, nn.Linear):
            macs[0] += m.in_features * m.out_features; acts[0] += o.numel()
    hs = [m.register_forward_hook(hook) for m in module.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        y = module(x)
    for h in hs:
        h.remove()
    return macs[0], acts[0], y


def nparams(*mods):
    return sum(p.numel() for m in mods for p in m.parameters())


def profile(state=None):
    net = MultiExitNet().eval()
    if state is not None:
        net.load_state_dict(state)
    torch.set_num_threads(2)
    out = {"constants": {"E_MAC_pJ": 4.6, "E_DRAM_word_pJ": 650, "E_SRAM_word_pJ": 5}, "ops": {}}
    for r in (32, 48):
        x = torch.randn(1, 3, r, r)
        m0, a0, f0 = count(net.stem, x)
        m1, a1, f1 = count(net.s1, f0); mh1, ah1, _ = count(net.h1, f1)
        m2, a2, f2 = count(net.s2, f1); mh2, ah2, _ = count(net.h2, f2)
        m3, a3, f3 = count(net.s3, f2); mh3, ah3, _ = count(net.h3, f3)
        stages = [(m0 + m1 + mh1, a0 + a1 + ah1, nparams(net.stem, net.s1, net.h1)),
                  (m2 + mh2, a2 + ah2, nparams(net.s2, net.h2)),
                  (m3 + mh3, a3 + ah3, nparams(net.s3, net.h3))]
        cm = ca = cp = 0
        for k in (1, 2, 3):
            cm += stages[k - 1][0]; ca += stages[k - 1][1]; cp += stages[k - 1][2]
            e = cm * E_MAC + cp * E_DRAM_WORD + 2 * ca * E_SRAM_WORD
            # latency: cascade up to exit k (all heads <= k evaluated)
            ts = []
            with torch.no_grad():
                for rep in range(60):
                    t0 = time.perf_counter(); net(x, upto=k); ts.append(time.perf_counter() - t0)
            out["ops"][f"{k}_{r}"] = {"MACs": int(cm), "params": int(cp), "activations": int(ca),
                                      "energy_mJ": e * 1e3, "E_compute_mJ": cm * E_MAC * 1e3,
                                      "E_weights_mJ": cp * E_DRAM_WORD * 1e3,
                                      "E_act_mJ": 2 * ca * E_SRAM_WORD * 1e3,
                                      "latency_ms": float(np.median(ts[10:]) * 1e3)}
    return out


if __name__ == "__main__":
    st = torch.load("runs/model_s0.pt")
    prof = profile(st)
    json.dump(prof, open("runs/energy_profile.json", "w"), indent=1)
    for k, v in prof["ops"].items():
        print(k, v)
