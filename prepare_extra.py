"""Decode the official GTSRB test images (frame-level normalisation / validation pool)
and the Indian traffic-sign images (cross-region evaluation) at 48x48."""
import io, sys, zipfile, numpy as np, pandas as pd, pyarrow.parquet as pq
from PIL import Image

D = sys.argv[1] if len(sys.argv) > 1 else "data"
S = 48
# ---- GTSRB official test set (independent recordings, no track ids) ----
csv = pd.read_csv(f"{D}/gtsrb_Test.csv")
tab = pq.read_table(f"{D}/gtsrb_test-00000-of-00001-747a54d4a6461a97.parquet")
meta = tab.select(["Width", "ClassId"]).to_pandas()
assert (meta.Width.values == csv.Width.values).all() and (meta.ClassId.values == csv.ClassId.values).all()
col = tab.column("Path"); x = np.zeros((len(csv), S, S, 3), np.uint8)
for i in range(len(csv)):
    x[i] = np.asarray(Image.open(io.BytesIO(col[i].as_py()["bytes"])).convert("RGB").resize((S, S), Image.BILINEAR))
np.savez_compressed(f"{D}/gtsrb_testpool.npz", x=x, y=csv.ClassId.values.astype(np.int64),
                    roi_w=(csv["Roi.X2"] - csv["Roi.X1"]).values.astype(np.float32))
print("test pool", x.shape)
# ---- Indian signs: semantic mapping to GTSRB ids (-1 = no counterpart, None = ambiguous, dropped) ----
MAP = {"SPEED_LIMIT_20": 0, "SPEED_LIMIT_30": 1, "SPEED_LIMIT_50": 2, "SPEED_LIMIT_60": 3, "SPEED_LIMIT_70": 4,
       "SPEED_LIMIT_80": 5, "OVERTAKING_PROHIBITED": 9, "TRUCK_PROHIBITED": 16, "GIVE_WAY": 13, "STOP": 14,
       "NO_ENTRY": 17, "LEFT_HAND_CURVE": 19, "RIGHT_HAND_CURVE": 20, "HUMP_OR_ROUGH_ROAD": 22, "SLIPPERY_ROAD": 23,
       "MEN_AT_WORK": 25, "TRAFFIC_SIGNAL": 26, "PEDESTRIAN_CROSSING": 27, "SCHOOL_AHEAD": 28, "CYCLE_CROSSING": 29,
       "COMPULSARY_TURN_RIGHT_AHEAD": 33, "COMPULSARY_TURN_LEFT_AHEAD": 34, "COMPULSARY_AHEAD": 35,
       "COMPULSARY_AHEAD_OR_TURN_RIGHT": 36, "COMPULSARY_AHEAD_OR_TURN_LEFT": 37, "COMPULSARY_KEEP_RIGHT": 38,
       "COMPULSARY_KEEP_LEFT": 39, "RESTRICTION_ENDS": 32,
       "ROUNDABOUT": None, "COMPULSARY_TURN_LEFT": None, "COMPULSARY_TURN_RIGHT": None, "TURN_RIGHT": None,
       "ALL_MOTOR_VEHICLE_PROHIBITED": None, "DIRECTION": None}
z = zipfile.ZipFile(f"{D}/indian.zip")
names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".png", ".jpeg"))]
xs, ys, cls = [], [], []
for n in names:
    c = n.split("/")[2]
    g = MAP.get(c, -1)
    if g is None:
        continue
    try:
        im = Image.open(io.BytesIO(z.read(n))).convert("RGB").resize((S, S), Image.BILINEAR)
    except Exception:
        continue
    xs.append(np.asarray(im)); ys.append(g); cls.append(c)
np.savez_compressed(f"{D}/indian.npz", x=np.stack(xs), y=np.array(ys, np.int64), cls=np.array(cls))
print("indian", len(xs), "mapped", int((np.array(ys) >= 0).sum()))
