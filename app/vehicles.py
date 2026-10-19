"""整台車輛的追蹤、車號延續, 以及外觀樣本收集 (供日後車輛外觀辨識使用)."""
import itertools
import json
import logging
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from vehicle import CLASS_NAMES

log = logging.getLogger("vehicles")
TYPE_ZH = {"car": "汽車", "motorcycle": "機車", "bus": "公車", "truck": "貨車"}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def contains(box, pt):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def inside(a, b):
    """框 a 落在框 b 內的面積比例."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1.0, (a[2] - a[0]) * (a[3] - a[1]))


def zone_overlap(box, zone):
    """車框落在圍籬內的面積比例 (圍籬須為凸多邊形). 無圍籬時為 1."""
    if zone is None:
        return 1.0
    x1, y1, x2, y2 = box
    rect = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    area, _ = cv2.intersectConvexConvex(rect, zone.astype(np.float32))
    return float(area) / max(1.0, (x2 - x1) * (y2 - y1))


class VehicleTrack:
    _ids = itertools.count(1)

    def __init__(self, box, conf, cls, now, startup):
        self.id = next(self._ids)
        self.box, self.conf, self.cls = box, conf, cls
        self.still_box, self.still_since = box, now
        self.first_seen = self.last_seen = now
        self.startup = startup        # 服務啟動時就已存在 (不發「無法辨識」通知)
        self.in_zone = False
        self.parked_since = None
        # 車號 (由車牌追蹤延續而來; 停妥後車牌被擋住仍保留)
        self.plate, self.plate_conf, self.plate_entry, self.plate_reads = None, 0.0, None, 0
        self.notified_plate = None
        self.unidentified_notified = False
        self.last_sample, self.samples = 0.0, 0
        self.arrival_done = False     # 進車通知已處理
        self.plate_lookup_done = False  # 已查過駛入過程中讀到的車牌
        self.plate_source = None        # 車號來自駛入過程中的讀取紀錄 (RecentPlates 項目)
        self.hits = 1                 # 被偵測到的次數 (排除一閃即逝的誤判)

    @property
    def type_name(self):
        return CLASS_NAMES.get(self.cls, "vehicle")

    def set_plate(self, plate, conf, entry, reads):
        if self.plate is None or conf > self.plate_conf:
            if self.plate and self.plate != plate:
                log.info("車輛 #%d 車號更新 %s -> %s", self.id, self.plate, plate)
            self.plate, self.plate_conf, self.plate_entry, self.plate_reads = plate, conf, entry, reads


class VehicleTracker:
    def __init__(self, lost_seconds=60, relink_seconds=600, still_ratio=0.1):
        self.tracks, self.lost = [], []
        self.lost_seconds, self.relink_seconds, self.still_ratio = lost_seconds, relink_seconds, still_ratio

    def update(self, dets, now, startup):
        """dets: [(x1, y1, x2, y2, conf, cls)]. 以 IoU 配對; 消失不久又出現在同位置的車視為同一台."""
        unmatched = list(range(len(dets)))
        for t in sorted(self.tracks, key=lambda t: -t.conf):
            best, best_iou = None, 0.3
            for i in unmatched:
                v = iou(t.box, dets[i][:4])
                if v > best_iou:
                    best, best_iou = i, v
            if best is not None:
                self._hit(t, dets[best], now)
                unmatched.remove(best)
        for i in unmatched:
            d = dets[i]
            revived = next((t for t in self.lost if iou(t.box, d[:4]) > 0.5), None)
            if revived:
                self.lost.remove(revived)
                self._hit(revived, d, now)
                self.tracks.append(revived)
            else:
                self.tracks.append(VehicleTrack(d[:4], d[4], d[5], now, startup))
        alive = []
        for t in self.tracks:
            (alive if now - t.last_seen < self.lost_seconds else self.lost).append(t)
        self.tracks = alive
        self.lost = [t for t in self.lost if now - t.last_seen < self.relink_seconds]

    def _hit(self, t, d, now):
        t.box, t.conf, t.cls, t.last_seen = d[:4], d[4], d[5], now
        t.hits += 1
        (cx, cy), (sx, sy) = ((t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2), \
            ((t.still_box[0] + t.still_box[2]) / 2, (t.still_box[1] + t.still_box[3]) / 2)
        if ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5 > self.still_ratio * max(1.0, t.box[2] - t.box[0]):
            t.still_box, t.still_since, t.parked_since = t.box, now, None

    def owner_of(self, plate_box):
        """車牌中心落在哪台車的框內 (多台時取最小的框)."""
        pt = ((plate_box[0] + plate_box[2]) / 2, (plate_box[1] + plate_box[3]) / 2)
        cands = [t for t in self.tracks if contains(t.box, pt)]
        return min(cands, key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1])) if cands else None


class Gallery:
    """外觀樣本: data/gallery/<車號>/ (白名單), _others/<車號>/ (非白名單), _unknown/ (無法辨識)."""

    def __init__(self, root, max_per_plate=300):
        self.root, self.max_per_plate = Path(root), max_per_plate
        self.lock = threading.Lock()

    def save(self, frame, box, camera, plate=None, whitelisted=False, conf=0.0, kind=""):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        mx, my = (x2 - x1) * 0.05, (y2 - y1) * 0.05
        crop = frame[int(max(0, y1 - my)):int(min(h, y2 + my)), int(max(0, x1 - mx)):int(min(w, x2 + mx))]
        if crop.size == 0:
            return None
        sub = plate if (plate and whitelisted) else f"_others/{plate}" if plate else "_unknown"
        d = self.root / sub
        stamp = datetime.now()
        # 紅外線夜間畫面為灰階 (三通道幾乎相同)
        night = float(np.abs(crop[::4, ::4, 0].astype(int) - crop[::4, ::4, 2]).mean()) < 3
        name = f"{stamp:%Y%m%d_%H%M%S}_{camera}{'_night' if night else ''}.jpg"
        with self.lock:
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            files = sorted(d.glob("*.jpg"))
            for old in files[:max(0, len(files) - self.max_per_plate)]:
                old.unlink(missing_ok=True)
            rec = {"time": stamp.isoformat(timespec="seconds"), "file": f"{sub}/{name}", "camera": camera,
                   "plate": plate, "whitelisted": whitelisted, "plate_conf": round(conf, 3), "type": kind,
                   "night": night, "box": [int(v) for v in box], "frame_size": [w, h]}
            with open(self.root / "index.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return sub + "/" + name
