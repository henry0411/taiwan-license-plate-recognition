"""Motorcycle appearance identification experiments (runs inside lpr:cpu).

Data
  /data/appearance/<name>/ref_*.jpg  reference crops chosen on 2026-10-28 (10 per motorcycle)
  /data/appearance/<name>/arr_*.jpg  5 arrival crops added on 2026-11-05 (reference augmentation)
  /data/appearance/_arrivals/*.jpg   crops of real arrivals, labelled by the annotator (arrival_labels.json)
  /data/apt_work/crops/*.jpg         379 labelled crops from 2026-10-23..27, 363 usable (crop_labels.json; labels by a single AI annotator)

Protocol (temporal, no tuning on the test set)
  E1: references = the 40 original refs, test = all 44 later arrivals.
  E2: reference augmentation: test = arrivals after 2026-11-05 only (never used as refs);
      compare original refs vs original + 5 augmented refs.
  E3: feature ablation on E1 with the deployed nearest-reference rule.
Output: /exp/appearance_results.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, "/app")
from appearance import MEAN, STD  # noqa: E402

NAME2ID = {"motorcycle_A": "A", "motorcycle_B": "B", "motorcycle_C": "C", "motorcycle_D": "D"}  # reference folder names
LAB2ID = {"A": "A", "B": "B", "X": "C", "R": "D"}
MIN_SIM, MIN_MARGIN = 0.70, 0.06

o = ort.SessionOptions()
o.intra_op_num_threads = 2
sess = ort.InferenceSession("/data/models/dinov2-small.onnx", sess_options=o, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name


def square(img, fill=114):
    h, w = img.shape[:2]
    s = max(h, w)
    sq = np.full((s, s) + img.shape[2:], fill, np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return sq


def dino(img, gray=True):
    if gray:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x = cv2.resize(square(g), (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = np.repeat(x[:, :, None], 3, 2)
    else:
        x = cv2.resize(square(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (224, 224),
                       interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    out = sess.run(None, {inp: x.transpose(2, 0, 1)[None]})[0][0]
    return out[0], out[1:].mean(0)


def norm(v):
    v = np.asarray(v, np.float32).ravel()
    return v / (np.linalg.norm(v) + 1e-9)


def hog(g, cell=16, bins=9):
    """Plain HOG: unsigned gradient orientation histograms on cell x cell blocks, each cell L2-normalised."""
    g = g.astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag, ang = np.hypot(gx, gy), np.rad2deg(np.arctan2(gy, gx)) % 180
    idx = np.minimum((ang / (180 / bins)).astype(int), bins - 1)
    n = g.shape[0] // cell
    out = np.zeros((n, n, bins), np.float32)
    for i in range(n):
        for j in range(n):
            sl = (slice(i * cell, (i + 1) * cell), slice(j * cell, (j + 1) * cell))
            out[i, j] = np.bincount(idx[sl].ravel(), mag[sl].ravel(), bins)
    return out / (np.linalg.norm(out, axis=2, keepdims=True) + 1e-6)


def features(img):
    cls_g, mean_g = dino(img, True)
    cls_c, mean_c = dino(img, False)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    g = cv2.resize(square(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), (128, 128), interpolation=cv2.INTER_AREA)
    raw = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    return {
        "dino_gray_cls+mean (deployed)": norm(np.concatenate([cls_g, mean_g])),
        "dino_gray_cls": norm(cls_g),
        "dino_gray_patchmean": norm(mean_g),
        "dino_rgb_cls+mean": norm(np.concatenate([cls_c, mean_c])),
        "hsv_histogram": norm(np.sqrt(hist)),
        "hog": norm(hog(g)),
        "raw_pixels_32": norm(raw - raw.mean()),
    }


def load(path):
    img = cv2.imread(str(path))
    assert img is not None, path
    return features(img)


root = Path("/data/appearance")
refs = {"orig": [], "aug": []}
for d in root.iterdir():
    if d.name.startswith("_") or not d.is_dir():
        continue
    for f in sorted(d.glob("*.jpg")):
        refs["aug" if f.name.startswith("arr_") else "orig"].append((NAME2ID[d.name], load(f)))

arr_lab = json.load(open("/exp/app_label/arrival_labels.json"))
arrivals = [(name, LAB2ID[lab], load(root / "_arrivals" / name)) for name, lab in sorted(arr_lab.items())]
crop_lab = json.load(open("/exp/app_label/crop_labels.json"))
crops = [(k, LAB2ID[v], load(Path(f"/data/apt_work/crops/{k}.jpg"))) for k, v in sorted(crop_lab.items()) if v != "U"]
METHODS = list(arrivals[0][2])


def rank(f, ref_list, method):
    best = {}
    for cid, feat in ref_list:
        s = float(feat[method] @ f[method])
        best[cid] = max(best.get(cid, -1.0), s)
    return sorted(best.items(), key=lambda kv: -kv[1])


def evaluate(tests, ref_list, method, rule=True):
    rows = []
    for name, truth, f in tests:
        r = rank(f, ref_list, method)
        top, sim, margin = r[0][0], r[0][1], r[0][1] - r[1][1]
        named = sim >= MIN_SIM and margin >= MIN_MARGIN
        rows.append({"name": name, "truth": truth, "top1": top, "sim": round(sim, 3), "margin": round(margin, 3),
                     "named": named, "second": r[1][0]})
    n = len(rows)
    named = [r for r in rows if r["named"]]
    return {
        "n": n,
        "top1_correct": sum(r["top1"] == r["truth"] for r in rows),
        "named": len(named),
        "named_correct": sum(r["top1"] == r["truth"] for r in named),
        "rows": rows,
    }


res = {"n_refs_orig": len(refs["orig"]), "n_refs_aug": len(refs["aug"]), "n_arrivals": len(arrivals),
       "n_crops": len(crops), "E1": {}, "E3_crops": {}}
for m in METHODS:
    res["E1"][m] = evaluate(arrivals, refs["orig"], m)
    # supplementary: the 379 earlier crops (same days the refs came from -> optimistic, refs excluded)
    ref_ids = {f"{int(p.stem[4:]):03d}" for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")
               for p in d.glob("ref_*.jpg")}
    res["E3_crops"][m] = evaluate([c for c in crops if c[0] not in ref_ids], refs["orig"], m)
    res["E3_crops"][m].pop("rows")

late = [a for a in arrivals if a[0][:8] >= "20261106"]
aug_names = {"20261030_210618", "20261031_062018", "20261103_111745", "20261030_201108", "20261031_125650"}
assert not any(a[0][:15] in aug_names for a in late)
res["E2"] = {"test_n": len(late),
             "orig": evaluate(late, refs["orig"], METHODS[0]),
             "aug": evaluate(late, refs["orig"] + refs["aug"], METHODS[0])}
json.dump(res, open("/exp/appearance_results.json", "w"), ensure_ascii=False, indent=1)
for m in METHODS:
    e, c = res["E1"][m], res["E3_crops"][m]
    print(f"{m:32s} arrivals top1 {e['top1_correct']}/{e['n']}  named {e['named_correct']}/{e['named']}"
          f"  | crops top1 {c['top1_correct']}/{c['n']}")
for k in ("orig", "aug"):
    e = res["E2"][k]
    print(f"E2 {k}: late arrivals top1 {e['top1_correct']}/{e['n']} named {e['named_correct']}/{e['named']}")
