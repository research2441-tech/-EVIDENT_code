# EVIDENT — reproducible implementation

Energy-budgeted, track-level traffic-sign recognition with policy-agnostic time-uniform conformal guarantees
and a typicality channel. Everything runs on a CPU (no GPU needed); a full run of five seeds takes about
3–4 hours on a 2-core machine.

## Files (single flat folder)
| File | Purpose |
|---|---|
| `download_data.sh` | downloads GTSRB (Hugging Face mirror `bazyl/GTSRB`) and the Indian sign set (`kannanwisen/Indian-Traffic-Sign-Classification`, CC BY 4.0) into `data/` |
| `prepare_data.py` | decodes GTSRB training crops into track-indexed arrays (track id, frame index, sign width) |
| `prepare_extra.py` | decodes the official GTSRB test crops and the Indian signs (with the semantic mapping to GTSRB classes) |
| `precompute_shift.py` | severity-3 corruptions (night, haze, motion blur, JPEG) and synthetic approach tracks |
| `common.py` | multi-exit backbone, operating points, track splits, augmentation, corruptions, synthetic tracks |
| `train.py` | trains the backbone for one seed and dumps per-frame log-probabilities for all operating points |
| `features.py` | class-conditional Mahalanobis distances for the typicality channel |
| `energy.py` | MAC / parameter / activation counts, modelled energy and measured CPU latency per operating point |
| `evident.py` | normalisers, e-values, envelope calibration, the EVIDENT controller and all baselines |
| `experiments.py` | main comparison, shifts, open-world, factorial ablation, sensitivity, re-partition validity |
| `stats.py` | aggregation over seeds, Wilcoxon / McNemar tests with Holm correction, Clopper–Pearson limits |
| `layout.py`, `diagrams.py`, `plots.py` | grayscale figures (diagram engine refuses overlapping boxes and labels) |
| `run_all.sh` | end-to-end pipeline |
| `requirements.txt` | Python dependencies |

## Run
```bash
pip install -r requirements.txt
bash run_all.sh            # data -> training (5 seeds) -> typicality -> energy -> experiments -> stats -> figures
```
Outputs: `runs/results.json`, `runs/stats.json`, `runs/pertrack_s*.npz`, figures in `out/`.

## Key settings
alpha_reg = 0.02, alpha_warn = 0.05, alpha_typ = 0.05, kappa = 0.02, kappa_typ = 0.1, confirmation a = 2,
deadline widths 40 / 50 px, non-commitment budget beta = dense non-commitment on calibration tracks + delta (delta = 0.02), admissible operating points: exits 2–3 at 32 / 48 px.

## Notes
* Splits are by track (one physical sign per track); never split by frame.
* Energy is modelled (45-nm per-operation energies, Horowitz, ISSCC 2014); latency is measured on the host CPU.
* The guarantee assumes calibration and test tracks of known signs are exchangeable within a safety group.
