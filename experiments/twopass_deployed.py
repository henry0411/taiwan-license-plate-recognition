"""Re-check the second detection pass with the DEPLOYED outdoor vehicle detector (yolov8n 640, pad 0, conf 0.2).
For every labelled readable plate, record whether the full-frame pass and/or the deployed second pass finds it."""
import json, sys
import cv2, numpy as np, onnxruntime as ort
sys.path.insert(0, "/app")
from fast_alpr.default_detector import DefaultDetector
from main import moved
from vehicle import VehicleDetector
def opts():
    o = ort.SessionOptions(); o.intra_op_num_threads = 2; return o
det = DefaultDetector("yolo-v9-t-384-license-plate-end2end", 0.25, ["CPUExecutionProvider"], opts())
veh = {"outside": VehicleDetector("/models/yolov8n.onnx", 0.2, sess_options=opts(), pad=0.0),
       "garage": VehicleDetector("/models/yolov8n.onnx", 0.12, sess_options=opts(), pad=0.0)}
meta = json.load(open("/exp/plates/meta.json"))
gt = {int(k): v for k, v in json.load(open("/exp/plate_label/plate_gt.json")).items()}
out = {}
by_img = {}
for i, g in gt.items():
    by_img.setdefault(meta[i]["image"], []).append(i)
for img_name, ids in by_img.items():
    frame = cv2.imread("/data/raw/" + img_name); h, w = frame.shape[:2]
    cam = meta[ids[0]]["camera"]
    full = [((d.bounding_box.x1, d.bounding_box.y1, d.bounding_box.x2, d.bounding_box.y2), d.confidence) for d in det.predict(frame)]
    second = []
    for x1, y1, x2, y2, vc, vk in veh[cam].detect(frame):
        if vc < (0.2 if cam == "outside" else 0.12):
            continue
        if (x2 - x1) * (y2 - y1) < (0.01 if cam == "outside" else 0.04) * w * h or (x2 - x1 > 0.6 * w and y2 - y1 > 0.6 * h):
            continue
        px, py = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        cx1, cy1, cx2, cy2 = int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, y2 + py))
        if cx2 - cx1 < 32 or cy2 - cy1 < 32:
            continue
        for d in det.predict(frame[cy1:cy2, cx1:cx2]):
            bb = d.bounding_box
            second.append(((bb.x1 + cx1, bb.y1 + cy1, bb.x2 + cx1, bb.y2 + cy1), d.confidence))
    for i in ids:
        b = meta[i]["box"]
        out[i] = {"full": any(moved(b, o) < 0.5 and c >= 0.4 for o, c in full),
                  "second_deployed": any(moved(b, o) < 0.5 and c >= 0.4 for o, c in second)}
json.dump(out, open("/exp/twopass_deployed.json", "w"))
import collections
c = collections.Counter((v["full"], v["second_deployed"]) for i, v in out.items() if gt[i]["tier"])
print({f"full={a},second={b}": n for (a, b), n in c.items()})
