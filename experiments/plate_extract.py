"""Extract plate crops + per-model / per-crop OCR probabilities from saved raw frames.

Runs inside the lpr:cpu image:  /app = deployed code (ro), /data = data (ro), /exp = output.
Output: /exp/plates/meta.json, /exp/plates/probs.npz (P[i, model, pad, slot, vocab]),
        /exp/plates/crops/<i>.jpg (enlarged crop used for manual labelling).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, "/app")
from fast_alpr.default_detector import DefaultDetector  # noqa: E402
from fast_plate_ocr.inference.plate_recognizer import _load_image_from_source, preprocess_image  # noqa: E402

from main import moved  # noqa: E402
from ocr import PADS, PlateOCR  # noqa: E402
from vehicle import VehicleDetector  # noqa: E402

cv2.setNumThreads(2)
OUT = Path("/exp/plates")
(OUT / "crops").mkdir(parents=True, exist_ok=True)


def opts():
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = 2, 1
    return o


det = DefaultDetector("yolo-v9-t-384-license-plate-end2end", 0.25, ["CPUExecutionProvider"], opts())
ocr = PlateOCR(["cct-xs-v2-global-model", "cct-s-v2-global-model"], sess_options=opts())
veh = {"outside": VehicleDetector("/models/yolov8n-416.onnx", 0.2, sess_options=opts(), pad=0.15),
       "garage": VehicleDetector("/models/yolov8n.onnx", 0.12, sess_options=opts(), pad=0.0)}


def plate_probs(frame, box):
    """(models, pads, slots, vocab)."""
    crops = list(ocr._crops(frame, box))
    if len(crops) != len(PADS):
        return None
    out = []
    for r in ocr.recs:
        conv = cv2.COLOR_BGR2GRAY if r.config.image_color_mode == "grayscale" else cv2.COLOR_BGR2RGB
        x = preprocess_image(_load_image_from_source([cv2.cvtColor(c, conv) for c in crops], r.config))
        y = r.model.run([r.plate_output_name], {"input": x})[0]
        out.append(y.reshape(len(crops), ocr.slots, -1))
    return np.stack(out)


meta, probs = [], []
files = sorted(Path("/data/raw").glob("*.jpg"))
for n, f in enumerate(files):
    frame = cv2.imread(str(f))
    if frame is None:
        continue
    h, w = frame.shape[:2]
    cam = "garage" if "_garage_" in f.name else "outside"
    full = [((d.bounding_box.x1, d.bounding_box.y1, d.bounding_box.x2, d.bounding_box.y2), float(d.confidence))
            for d in det.predict(frame)]
    boxes = [(b, c, "full") for b, c in full]
    for x1, y1, x2, y2, vc, vk in veh[cam].detect(frame):
        if (x2 - x1) * (y2 - y1) < 0.01 * w * h or (x2 - x1 > 0.6 * w and y2 - y1 > 0.6 * h):
            continue
        px, py = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        cx1, cy1, cx2, cy2 = int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, y2 + py))
        if cx2 - cx1 < 32 or cy2 - cy1 < 32:
            continue
        for d in det.predict(frame[cy1:cy2, cx1:cx2]):
            bb = d.bounding_box
            b = (bb.x1 + cx1, bb.y1 + cy1, bb.x2 + cx1, bb.y2 + cy1)
            hit = next((i for i, (o, _, _) in enumerate(boxes) if moved(b, o) < 0.5), None)
            if hit is None:
                boxes.append((b, float(d.confidence), "vehicle"))
            elif boxes[hit][2] == "full":
                boxes[hit] = (boxes[hit][0], boxes[hit][1], "both")
    for b, c, src in boxes:
        p = plate_probs(frame, b)
        if p is None:
            continue
        i = len(meta)
        x1, y1, x2, y2 = map(int, b)
        dx, dy = int((x2 - x1) * 0.15), int((y2 - y1) * 0.3)
        crop = frame[max(0, y1 - dy):min(h, y2 + dy), max(0, x1 - dx):min(w, x2 + dx)]
        s = 72 / max(1, crop.shape[0])
        cv2.imwrite(str(OUT / "crops" / f"{i}.jpg"), cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC))
        meta.append({"i": i, "image": f.name, "camera": cam, "box": [round(v, 1) for v in b], "det_conf": round(c, 3),
                     "source": src, "plate_w": round(b[2] - b[0], 1), "frame_w": w,
                     "gray": bool(np.abs(frame[..., 0].astype(int) - frame[..., 2]).mean() < 2)})
        probs.append(p.astype(np.float16))
    if n % 50 == 0:
        print(n, len(files), len(meta), flush=True)

json.dump(meta, open(OUT / "meta.json", "w"))
np.savez_compressed(OUT / "probs.npz", P=np.stack(probs))
print("done", len(meta), "alphabet", ocr.alphabet, "pad", ocr.pad_char, "slots", ocr.slots)
json.dump({"alphabet": ocr.alphabet, "pad": ocr.pad_char, "slots": ocr.slots}, open(OUT / "ocr_info.json", "w"))
