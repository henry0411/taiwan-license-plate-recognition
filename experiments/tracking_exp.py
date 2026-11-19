"""Parked-state hysteresis ablation: replay the deployed vehicle detector + tracker on saved garage frames.

Runs inside lpr:cpu. /data/motion/garage holds the last 1500 frames saved when the garage camera saw motion
(the deployment saves at most one frame per second while something moves).
Step 1 caches per-frame detections (/exp/tracking_dets.json), step 2 replays the tracker with
different parked-state thresholds and counts "parked" (停妥) events per vehicle track.
"""
import collections
import datetime
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, "/app")
from main import inside  # noqa: E402
from vehicle import VehicleDetector  # noqa: E402
from vehicles import VehicleTracker  # noqa: E402

cache = Path("/exp/tracking_dets.json")
if not cache.exists():
    o = ort.SessionOptions()
    o.intra_op_num_threads = 2
    det = VehicleDetector("/models/yolov8n.onnx", 0.12, sess_options=o, pad=0.0)
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
    frames = []
    for f in sorted(Path("/data/motion/garage").glob("*.jpg")):
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        ds = [d for d in det.detect(img) if area(d) >= 0.04 * w * h]
        ds = [d for d in ds if sum(inside(e[:4], d[:4]) >= 0.7 for e in ds if e is not d) < 2]
        ds = [d for d in ds if not any(e is not d and e[5] == d[5] and e[4] > d[4] and inside(d[:4], e[:4]) >= 0.8
                                       for e in ds)]
        ds = [d for d in ds if not any(e is not d and e[4] > d[4] and inside(e[:4], d[:4]) >= 0.8
                                       and area(d) >= 1.5 * area(e) for e in ds)]
        ts = datetime.datetime.strptime(f.stem, "%Y%m%d_%H%M%S").timestamp()
        frames.append((ts, f.stem, [list(map(float, d)) for d in ds]))
    json.dump({"w": w, "h": h, "frames": frames}, open(cache, "w"))
D = json.load(open(cache))
F = D["frames"]


def replay(parked_ratio, dwell=3, notify_after=120):
    tr = VehicleTracker(60, 600, parked_ratio=parked_ratio)
    parked, per_track, reach, jumps = 0, collections.Counter(), 0, []
    reached = set()
    for ts, _, ds in F:
        tr.update([tuple(d) for d in ds], ts, False)
        for v in tr.tracks:
            if ts - v.still_since < dwell:
                continue
            if v.parked_since is None:
                v.parked_since = ts
                parked += 1
                per_track[v.id] += 1
            if v.id not in reached and v.last_seen == ts and v.hits >= 5 and ts - v.parked_since >= notify_after:
                reached.add(v.id)
    return {"parked_events": parked, "tracks": len(per_track), "max_per_track": max(per_track.values() or [0]),
            "tracks_reaching_notify": len(reached)}


def jitter():
    """Centre displacement (in box widths) of parked tracks between consecutive detections, deployed tracker."""
    tr = VehicleTracker(60, 600)
    out = []
    orig = tr._hit

    def hit(t, d, now):
        if t.parked_since is not None:
            (cx, cy), (sx, sy) = ((d[0] + d[2]) / 2, (d[1] + d[3]) / 2), \
                ((t.still_box[0] + t.still_box[2]) / 2, (t.still_box[1] + t.still_box[3]) / 2)
            out.append(((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5 / max(1.0, d[2] - d[0]))
        orig(t, d, now)
    tr._hit = hit
    for ts, _, ds in F:
        tr.update([tuple(d) for d in ds], ts, False)
        for v in tr.tracks:
            if ts - v.still_since >= 3 and v.parked_since is None:
                v.parked_since = ts
    a = np.array(out)
    return {"n": len(a), "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)), "share_gt_0.1": float((a > 0.1).mean()),
            "share_gt_0.3": float((a > 0.3).mean())}


res = {"frames": len(F), "first": F[0][1], "last": F[-1][1],
       "frames_with_vehicle": sum(1 for f in F if f[2]),
       "jitter": jitter(),
       "sweep": {str(r): replay(r) for r in (None, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0)}}
json.dump(res, open("/exp/tracking_results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
