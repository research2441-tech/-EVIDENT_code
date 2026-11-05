"""Decode GTSRB (HF mirror of the official training set) into track-indexed arrays.

Each GTSRB training track holds ~30 frames of one physical sign recorded while the
vehicle approached it. Track ids come from the original file names
(ClassId_TrackId_FrameId). Images are stored at 48x48 RGB uint8 together with the
annotated sign width in pixels (used later as the distance proxy).
"""
import io, sys, numpy as np, pandas as pd, pyarrow.parquet as pq
from PIL import Image

D = sys.argv[1] if len(sys.argv) > 1 else "data"
S = 48
csv = pd.read_csv(f"{D}/gtsrb_Train.csv")
tab = pq.read_table(f"{D}/gtsrb_train-00000-of-00001-dc762c064c221993.parquet")
pdf = tab.select(["Width", "Height", "ClassId"]).to_pandas()
assert len(pdf) == len(csv)
assert (pdf.Width.values == csv.Width.values).all() and (pdf.ClassId.values == csv.ClassId.values).all()
m = csv.Path.str.extract(r"/(\d+)_(\d+)_(\d+)")
csv["track"] = m[1].astype(int); csv["frame"] = m[2].astype(int)
imgs = np.zeros((len(csv), S, S, 3), np.uint8)
col = tab.column("Path")
for i in range(len(csv)):
    b = col[i].as_py()["bytes"]
    im = Image.open(io.BytesIO(b)).convert("RGB").resize((S, S), Image.BILINEAR)
    imgs[i] = np.asarray(im)
tid = (csv.ClassId.values * 1000 + csv.track.values).astype(np.int64)
roi_w = (csv["Roi.X2"] - csv["Roi.X1"]).values.astype(np.float32)
np.savez_compressed(f"{D}/gtsrb_tracks.npz", x=imgs, y=csv.ClassId.values.astype(np.int64),
                    tid=tid, frame=csv.frame.values.astype(np.int64), roi_w=roi_w,
                    w=csv.Width.values.astype(np.float32))
print("saved", imgs.shape, "tracks", len(np.unique(tid)))
