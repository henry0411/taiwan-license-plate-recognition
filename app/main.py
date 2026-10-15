"""台灣車牌辨識服務: MJPEG 鏡頭 -> 車牌偵測/追蹤 -> 電子圍籬/靜止判斷 -> OCR 投票 -> Synology Chat."""
import argparse
import json
import itertools
import logging
import os
import threading
import time
from datetime import datetime
from html import escape as html_escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import requests
import yaml
from types import SimpleNamespace

from fast_alpr.default_detector import DefaultDetector

from home import HomeCheck
from vehicle import VehicleDetector
from vehicles import TYPE_ZH, Gallery, VehicleTracker, inside, zone_overlap
from notify import SynologyChat
from ocr import PlateOCR
from tw_plate import Decoder, plain

log = logging.getLogger("lpr")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
SNAP_DIR = DATA_DIR / "snapshots"
RAW_DIR = DATA_DIR / "raw"
LIVE = {}  # 鏡頭名稱 -> 最新標註畫面 (JPEG bytes)
WORKERS = {}  # 鏡頭名稱 -> CameraWorker


# ---------------------------------------------------------------- MJPEG 讀取
class MjpegReader(threading.Thread):
    """逐張抓取 MJPEG 畫面: 連線 -> 讀到第一張完整 JPEG -> 斷線, 只保留最新一張 (需要時才解碼).

    高畫質串流每秒可達十幾 MB, 持續解析很耗 CPU; 本服務每秒只需約 1 張, 逐張抓取較省資源.
    """

    def __init__(self, name, url, min_interval=0.0):
        super().__init__(daemon=True, name=f"reader-{name}")
        self.name_, self.url, self.min_interval = name, url, min_interval
        self.latest, self.stamp = None, 0.0
        self.lock = threading.Lock()

    def grab(self):
        with requests.get(self.url, stream=True, timeout=(5, 15)) as r:
            r.raise_for_status()
            buf, s = bytearray(), -1
            for chunk in r.iter_content(chunk_size=262144):
                start = max(0, len(buf) - 1)
                buf += chunk
                if s < 0:
                    s = buf.find(b"\xff\xd8", start)
                    if s < 0:
                        continue
                    start = s + 2
                e = buf.find(b"\xff\xd9", max(start, s + 2))
                if e >= 0:
                    return bytes(buf[s:e + 2])
                if len(buf) > 16_000_000:
                    raise IOError("找不到完整 JPEG")
        raise IOError("串流結束")

    def run(self):
        ok = None
        while True:
            start = time.time()
            try:
                data = self.grab()
                with self.lock:
                    self.latest, self.stamp = data, time.time()
                if ok is not True:
                    log.info("[%s] 串流已連線", self.name_)
                ok = True
            except Exception as e:  # noqa: BLE001 - 斷線自動重連
                if ok is not False:
                    log.warning("[%s] 串流中斷: %s, 稍後重試", self.name_, e)
                ok = False
                time.sleep(5)
            time.sleep(max(0.0, self.min_interval - (time.time() - start)))

    def frame(self, max_age=10):
        with self.lock:
            data, stamp = self.latest, self.stamp
        if data is None or time.time() - stamp > max_age:
            return None
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- 車牌追蹤
def center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def moved(a, b):
    """兩個車牌框中心距離 / 車牌寬度."""
    (ax, ay), (bx, by) = center(a), center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / max(1.0, a[2] - a[0])


class Track:
    _ids = itertools.count(1)

    def __init__(self, box, now):
        self.id = next(self._ids)
        self.box = box
        self.still_box, self.still_since = box, now  # 靜止判斷基準
        self.last_seen = now
        self.last_ocr = 0.0
        self.prob_sum = None   # 累積的 OCR 機率矩陣 (多張畫面平均, 比單張投票穩定)
        self.reads = 0
        self.history = []      # 每次累積後的解碼結果
        self.notified = None
        self.last_read = None
        self.result = None     # (車牌, 信心, 白名單項目) 確認後
        self.attempts = 0      # OCR 嘗試次數 (含不採用的模糊畫面)
        self.vehicle = None    # 所屬車輛 (VehicleTrack)


# ---------------------------------------------------------------- 通知冷卻 (存檔, 重啟後保留)
class Cooldowns:
    def __init__(self, path):
        self.path, self.lock = Path(path), threading.Lock()
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            self.data = {}

    def hit(self, key, now, seconds):
        """未在冷卻中則記錄並回傳 True."""
        with self.lock:
            if now - self.data.get(key, -1e18) < seconds:
                return False
            self.data[key] = now
            self.data = {k: v for k, v in self.data.items() if now - v < 7 * 86400}
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.data), encoding="utf-8")
                tmp.replace(self.path)
            except OSError as e:
                log.warning("冷卻紀錄存檔失敗: %s", e)
            return True


# ---------------------------------------------------------------- 白名單
class Whitelist:
    """白名單 YAML, 檔案修改後自動重新載入 (不需重啟)."""

    def __init__(self, path):
        self.path = Path(path)
        self.mtime = None
        self.entries, self.match_ratio, self.notify_whitelisted = [], 0.85, True
        self.lock = threading.Lock()

    def _reload(self):
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self.entries = []
            return
        if mtime == self.mtime:
            return
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        entries = []
        for e in data.get("plates") or []:
            e = {"plate": e} if isinstance(e, str) else dict(e)
            e["plate"] = str(e["plate"]).upper().strip()
            entries.append(e)
        self.entries = entries
        self.match_ratio = float(data.get("match_ratio", 0.85))
        self.notify_whitelisted = bool(data.get("notify_whitelisted", True))
        self.mtime = mtime
        log.info("白名單已載入: %d 筆", len(entries))

    def best(self, probs, decoder):
        """回傳 (最可能的白名單項目, 其機率) 或 (None, 0)."""
        with self.lock:
            self._reload()
            best = (None, 0.0)
            for e in self.entries:
                like = decoder.likelihood(probs, e["plate"])
                if like > best[1]:
                    best = (e, like)
            return best

    def lookup(self, plate):
        with self.lock:
            self._reload()
            return next((e for e in self.entries if plain(e["plate"]) == plain(plate)), None)


# ---------------------------------------------------------------- 鏡頭處理
class CameraWorker(threading.Thread):
    def __init__(self, cam, cfg, models, chat, cooldowns, whitelist, gallery):
        super().__init__(daemon=True, name=f"cam-{cam['name']}")
        self.cam, self.cfg = cam, cfg
        self.name_ = cam["name"]
        self.label = cam.get("label") or self.name_   # 顯示名稱 (通知訊息 / 網頁)
        self.reader = MjpegReader(self.name_, cam["url"], 0.8 / cam.get("process_fps", 1))
        self.models, self.chat, self.whitelist, self.gallery = models, chat, whitelist, gallery
        self.cooldowns = cooldowns
        self.tracks = []        # 車牌追蹤
        vcfg = cfg["vehicle"]
        self.vcfg = vcfg
        self.vehicles = VehicleTracker(vcfg["lost_seconds"], vcfg["relink_seconds"])
        self.vehicle_det = models.vehicle(cam.get("vehicle_model", vcfg["model"]),
                                          cam.get("vehicle_pad", vcfg["pad"]))
        self.last_vehicle_run = 0.0
        self.prev_thumb, self.last_full = None, 0.0
        self.started = time.time()
        self.moto = cam.get("include_motorcycle_formats", cfg["include_motorcycle_formats"])
        self.zone_norm = np.array(cam["zone"], np.float32) if cam.get("zone") else None
        self.zone_px = None     # 本張畫面的圍籬 (像素座標, 旋轉鏡頭會依原位比對結果校正)
        # 排除區 (例如水溝蓋): 車框大部分落在其中的偵測結果直接忽略
        self.ignore_norm = [np.array(z, np.float32) for z in cam.get("ignore") or []]
        self.ignore_px = []
        self.parked_spots = []  # 進車通知: 最近有車停著的位置 [(cx, cy, 時間)] (畫面比例)
        self.motion_times = []  # 進車通知: 畫面有明顯局部變化 (人/車移動) 的時間
        self.last_good, self.offline_alerted = time.time(), False
        # 旋轉鏡頭: 只有轉回原位時才辨識
        self.home = HomeCheck(DATA_DIR / "home" / self.name_, cam.get("home_min_inliers", 20),
                              cam.get("home_max_shift", 0.08)) if cam.get("rotating") else None
        self.home_status = ""
        self.active = True
        WORKERS[self.name_] = self

    # 車牌中心點是否在圍籬內
    def in_zone(self, box, shape=None):
        if self.zone_px is None:
            return True
        cx, cy = center(box)
        return cv2.pointPolygonTest(self.zone_px, (float(cx), float(cy)), False) >= 0

    def update_zone(self, frame):
        """計算本張畫面的圍籬. 旋轉鏡頭不在原位時回傳 False (暫停辨識)."""
        h, w = frame.shape[:2]
        zone = self.zone_norm
        if self.home is not None:
            if not self.home.has_reference:
                self.home_status = "尚未設定原位參考畫面 - 暫停辨識"
                return False
            H, n = self.home.check(frame, time.time())
            if H is None:
                self.home_status = f"非原位 (比對 {n}) - 暫停辨識"
                return False
            self.home_status = f"原位 (比對 {n})"
            warp = lambda z: cv2.perspectiveTransform(z.reshape(-1, 1, 2), H).reshape(-1, 2)  # noqa: E731
        else:
            warp = lambda z: z  # noqa: E731
        self.zone_px = None if zone is None else (warp(zone) * [w, h]).astype(np.float32)
        self.ignore_px = [(warp(z) * [w, h]).astype(np.float32) for z in self.ignore_norm]
        return True

    def update_tracks(self, boxes, now):
        unmatched = list(range(len(boxes)))
        for t in self.tracks:
            best, best_d = None, 1.5
            for i in unmatched:
                d = moved(t.box, boxes[i])
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                t.box = boxes[best]
                t.last_seen = now
                unmatched.remove(best)
                if moved(t.box, t.still_box) > 0.5:  # 有移動 -> 重新計時
                    t.still_box, t.still_since = t.box, now
        for i in unmatched:
            self.tracks.append(Track(boxes[i], now))
        lost = self.cfg["track_lost_seconds"]
        self.tracks = [t for t in self.tracks if now - t.last_seen < lost]

    def read(self, frame, box):
        """單張畫面 OCR, 回傳 (機率矩陣, 台灣格式車牌 或 None, 信心)."""
        probs = self.models.ocr.probs(frame, box)
        if probs is None:
            return None, None, 0.0
        plate, conf = self.models.decoder.decode(probs, self.moto)
        return probs, plate, conf

    def judge(self, t):
        """依累積結果判斷是否確認. 回傳 (車牌, 信心, 白名單項目 或 None) 或 None."""
        mean = t.prob_sum / t.reads
        plate, conf = self.models.decoder.decode(mean, self.moto)
        t.history.append(plate)
        t.last_read = plate
        n = self.cfg["min_votes"]
        if t.reads < n:
            return None
        # 白名單: 與最佳解碼結果機率相近即視為該車 (容許夜間/模糊造成的個別字元誤讀)
        entry, like = self.whitelist.best(mean, self.models.decoder)
        if entry and like >= conf * self.whitelist.match_ratio and like >= self.cfg["ocr_min_conf"] * 0.8:
            return entry["plate"], like, entry
        if conf >= self.cfg["ocr_min_conf"] and len(set(t.history[-n:])) == 1:
            return plate, conf, None
        return None

    # ------------------------------------------------------------ 通知
    def _snapshot(self, frame, boxes, stamp, tag):
        """存未標註原圖與標註截圖, 回傳檔名. boxes: [(box, 文字, BGR 顏色)]."""
        fname = f"{stamp:%Y%m%d_%H%M%S}_{self.name_}_{tag}.jpg"
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(RAW_DIR / fname), frame)  # 未標註原圖, 供日後調校
        snap = frame.copy()
        lw = max(2, snap.shape[1] // 500)
        for box, label, color in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(snap, (x1, y1), (x2, y2), color, lw)
            if label:
                cv2.putText(snap, label, (max(0, x1), max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4 * lw, color, lw)
        SNAP_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(SNAP_DIR / fname), snap)
        return fname

    def check_offline(self, now):
        """超過 offline_alert 秒沒有正常畫面 (斷線 / 灰色「CONNECTION TO CAMERA LOST」畫面) 通知一次."""
        limit = self.cfg.get("camera_offline_alert", 600)
        if not limit or self.offline_alerted or now - self.last_good < limit:
            return
        self.offline_alerted = True
        log.warning("[%s] 鏡頭超過 %d 分鐘沒有畫面", self.name_, limit // 60)
        self._send_text(f"⚠️ 鏡頭斷線：{self.label}\n超過 {limit // 60} 分鐘沒有畫面，請檢查鏡頭電源/網路")

    @property
    def status(self):
        if self.offline_alerted:
            return "⚠️ 鏡頭斷線"
        return self.home_status

    def _send_text(self, text):
        threading.Thread(target=self.chat.send, args=(text, None), daemon=True).start()

    def _send(self, text, fname):
        base = self.cfg.get("snapshot_base_url")
        file_url = f"{base.rstrip('/')}/snapshots/{fname}" if base else None
        threading.Thread(target=self.chat.send, args=(text, file_url), daemon=True).start()

    def notify_plate(self, frame, plate, conf, entry, reads, now, plate_box=None, vehicle=None):
        key = f"{self.name_}|{plate}"
        if not self.cooldowns.hit(key, now, self.cfg["plate_cooldown_seconds"]):
            log.info("[%s] %s 冷卻中, 不重複通知", self.name_, plate)
            return
        stamp = datetime.now()
        boxes = []
        if vehicle is not None:
            boxes.append((vehicle.box, "", (255, 160, 0)))
        if plate_box is not None:
            boxes.append((plate_box, plate, (0, 255, 0)))
        elif vehicle is not None:
            boxes[0] = (vehicle.box, plate, (0, 255, 0))
        fname = self._snapshot(frame, boxes, stamp, plate)
        status = f"白名單 {entry.get('name') or ''}".strip() if entry else "非白名單"
        log.info("[%s] ✅ 確認車牌 %s [%s] (畫面數 %d, 信心 %.2f)%s -> %s", self.name_, plate, status, reads, conf,
                 " (車號延續, 停妥時車牌未入鏡)" if vehicle is not None and plate_box is None else "", fname)
        if entry and not self.whitelist.notify_whitelisted:
            return
        head = (f"✅ 白名單車輛：*{plate}*" + (f"（{entry['name']}）" if entry.get("name") else "")
                if entry else f"⚠️ 非白名單車輛：*{plate}*")
        lines = [head, f"📷 鏡頭：{self.label}", f"🕒 時間：{stamp:%Y-%m-%d %H:%M:%S}",
                 f"🎯 信心：{conf:.0%}（綜合 {reads} 張畫面）"]
        if vehicle is not None and plate_box is None:
            lines.append("ℹ️ 車牌於駛入時辨識，停妥後已不在畫面中")
        self._send("\n".join(lines), fname)

    def notify_unidentified(self, frame, v, now):
        # 同一位置 (畫面切成 8x8 格) 的無法辨識通知冷卻, 避免長期停放的車重複通知
        h, w = frame.shape[:2]
        cx, cy = center(v.box)
        key = f"{self.name_}|?|{int(8 * cx / w)},{int(8 * cy / h)}"
        if not self.cooldowns.hit(key, now, self.vcfg["unidentified_cooldown"]):
            log.info("[%s] 車輛 #%d 無法辨識 (同位置冷卻中)", self.name_, v.id)
            return
        stamp = datetime.now()
        fname = self._snapshot(frame, [(v.box, "?", (0, 0, 255))], stamp, "unidentified")
        self.gallery.save(frame, v.box, self.name_, kind=v.type_name)
        minutes = (now - v.parked_since) / 60
        log.info("[%s] ❓ 車輛 #%d (%s) 停妥 %.1f 分鐘仍無法辨識車牌 -> %s",
                 self.name_, v.id, v.type_name, minutes, fname)
        if not self.cam.get("notify_unidentified", True):
            return
        self._send("\n".join([
            "❓ 有車輛停入，但無法辨識車牌",
            f"🚗 類型：{TYPE_ZH.get(v.type_name, v.type_name)}",
            f"📷 鏡頭：{self.label}",
            f"🕒 時間：{stamp:%Y-%m-%d %H:%M:%S}（已停 {minutes:.0f} 分鐘）",
        ]), fname)

    # ------------------------------------------------------------ 主流程
    def process(self, frame, now):
        self.last_frame = frame
        self.active = self.update_zone(frame)
        if not self.active:
            if self.tracks or self.vehicles.tracks:
                log.info("[%s] 鏡頭離開原位, 暫停辨識", self.name_)
            self.tracks = []  # 轉回原位後重新追蹤 / 計算靜止時間
            self.vehicles = VehicleTracker(self.vcfg["lost_seconds"], self.vcfg["relink_seconds"])
            self.annotate(frame, now)
            return

        # 畫面幾乎沒變 (停車場多數時間如此) -> 略過偵測, 省 CPU; 每隔 static_refresh 秒仍完整處理一次
        thumb = cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        changed = 1.0 if self.prev_thumb is None else float((cv2.absdiff(thumb, self.prev_thumb) > 12).mean())
        # 局部變化才算有東西在動; 整個畫面都變 (日夜模式切換/開燈) 不算
        if 0.02 <= changed < 0.6 and self.prev_thumb is not None:
            self.motion_times = [t for t in self.motion_times if now - t < 900] + [now]
        if changed < self.cfg.get("static_change", 0.005) and now - self.last_full < self.cfg.get("static_refresh", 10):
            return
        self.prev_thumb, self.last_full = thumb, now

        # 車輛偵測 (較耗時, 依間隔執行)
        if now - self.last_vehicle_run >= self.vcfg["interval"]:
            self.last_vehicle_run = now
            self.update_vehicles(frame, now)
            vehicle_frame = True
        else:
            vehicle_frame = False

        dets = self.models.detector.predict(frame)
        boxes = [(d.bounding_box.x1, d.bounding_box.y1, d.bounding_box.x2, d.bounding_box.y2) for d in dets]
        self.update_tracks(boxes, now)
        dwell = self.cam.get("dwell_seconds", 10)
        for t in self.tracks:
            if t.last_seen != now:
                continue
            owner = self.vehicles.owner_of(t.box)
            if owner is not None:
                t.vehicle = owner
            v = t.vehicle if t.vehicle in self.vehicles.tracks else None
            if t.result:
                if v is not None:
                    v.set_plate(*t.result, t.reads)
                elif self.in_zone(t.box) and now - t.still_since >= dwell and t.notified is None and vehicle_frame:
                    # 偵測不到整台車 (例如被切掉太多): 以車牌本身停留判斷
                    t.notified = t.result[0]
                    self.notify_plate(frame, *t.result, t.reads, now, plate_box=t.box)
                continue
            # 車牌在圍籬內, 或所屬車輛在圍籬內 -> 駛入過程中就開始辨識 (不必等停妥)
            if not (self.in_zone(t.box) or (v is not None and v.in_zone)):
                continue
            # 嘗試多次仍無法確認 (例如角度太斜) -> 降低頻率, 省 CPU
            interval = (self.cfg["ocr_interval"] if t.attempts < self.cfg.get("ocr_max_attempts", 20)
                        else self.cfg.get("ocr_retry_interval", 30))
            if now - t.last_ocr < interval:
                continue
            t.last_ocr = now
            t.attempts += 1
            probs, plate, conf = self.read(frame, t.box)
            log.debug("[%s] #%d OCR -> %s (%.2f)", self.name_, t.id, plate, conf)
            if probs is None or conf < self.cfg["frame_min_conf"]:  # 模糊/遮擋的畫面不納入
                continue
            t.prob_sum = probs if t.prob_sum is None else t.prob_sum + probs
            t.reads += 1
            result = self.judge(t)
            if result:
                t.result = result
                log.info("[%s] 車牌 #%d 辨識為 %s (%.2f)%s", self.name_, t.id, result[0], result[1],
                         f" -> 車輛 #{v.id}" if v is not None else "")
                if v is not None:
                    v.set_plate(*result, t.reads)

        if vehicle_frame:
            self.check_vehicles(frame, now)
            if self.cam.get("notify_parked"):
                self.mark_parked_spots(frame, now)
        self.annotate(frame, now)

    def update_vehicles(self, frame, now):
        h, w = frame.shape[:2]
        min_area = self.cam.get("vehicle_min_area", self.vcfg["min_area"]) * w * h
        dets = [d for d in self.vehicle_det.detect(frame, self.cam.get("vehicle_classes"))
                if (d[2] - d[0]) * (d[3] - d[1]) >= min_area and d[4] >= self.cam.get("vehicle_conf", self.vcfg["conf"])
                and not any(zone_overlap(d[:4], z) >= 0.5 for z in self.ignore_px)]
        # 一個框同時包住兩台以上其他車輛 (把相鄰的車併成一台的誤判) -> 忽略
        dets = [d for d in dets if sum(inside(e[:4], d[:4]) >= 0.7 for e in dets if e is not d) < 2]
        # 小框幾乎落在另一台同類、信心較高的車框內 (同一台車的局部) -> 忽略小框
        dets = [d for d in dets if not any(e is not d and e[5] == d[5] and e[4] > d[4] and inside(d[:4], e[:4]) >= 0.8
                                           for e in dets)]
        # 一個大框幾乎包住另一台信心較高的車 (同一台車多框出一截) -> 忽略大框
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
        dets = [d for d in dets if not any(e is not d and e[4] > d[4] and inside(e[:4], d[:4]) >= 0.8
                                           and area(d) >= 1.5 * area(e) for e in dets)]
        startup = now - self.started < self.vcfg["startup_grace"]
        self.vehicles.update(dets, now, startup)
        for v in self.vehicles.tracks:
            if v.last_seen == now:
                v.in_zone = zone_overlap(v.box, self.zone_px) >= self.vcfg["zone_overlap"]

    def check_vehicles(self, frame, now):
        dwell = self.cam.get("dwell_seconds", 10)
        for v in self.vehicles.tracks:
            if not v.in_zone or now - v.still_since < dwell:
                continue
            if v.parked_since is None:
                v.parked_since = now
                log.info("[%s] 車輛 #%d (%s) 停妥%s", self.name_, v.id, v.type_name,
                         f", 車號 {v.plate}" if v.plate else "")
            if v.plate and v.notified_plate != v.plate:
                v.notified_plate = v.plate
                plate_box = next((t.box for t in self.tracks
                                  if t.last_seen == now and t.vehicle is v and t.result), None)
                self.notify_plate(frame, v.plate, v.plate_conf, v.plate_entry, v.plate_reads, now,
                                  plate_box=plate_box, vehicle=v)
            elif self.cam.get("notify_parked"):
                self.check_arrival(frame, v, now)
            elif (not v.plate and not v.unidentified_notified and not v.startup
                  and now - v.parked_since >= self.vcfg["unidentified_after"]):
                v.unidentified_notified = True
                self.notify_unidentified(frame, v, now)
            # 外觀樣本: 已知車號的停妥車輛, 定時存一張整車裁切圖
            if (v.plate and self.cfg["gallery"]["enabled"] and v.samples < self.cfg["gallery"]["max_per_track"]
                    and now - v.last_sample >= self.cfg["gallery"]["interval"]):
                v.last_sample = now
                v.samples += 1
                if v.plate_entry or self.cfg["gallery"]["save_others"]:
                    self.gallery.save(frame, v.box, self.name_, v.plate, v.plate_entry is not None,
                                      v.plate_conf, v.type_name)

    def check_arrival(self, frame, v, now):
        """進車通知: 停妥一段時間後傳照片. 同一位置最近一直有車 (同一台車偵測框跳動/重新追蹤) 不重複通知."""
        if v.arrival_done or v.last_seen != now:
            return
        h, w = frame.shape[:2]
        cx, cy = (v.box[0] + v.box[2]) / 2 / w, (v.box[1] + v.box[3]) / 2 / h
        rearm = self.cam.get("parked_rearm_seconds", 600)
        self.parked_spots = [s for s in self.parked_spots if now - s[2] < rearm]
        near = [i for i, (x, y, _) in enumerate(self.parked_spots) if ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 < 0.08]
        if near or v.startup or v.plate:
            v.arrival_done = True    # 原本就停著的車 / 已讀到車牌 (車牌通知已附照片)
        elif now - v.parked_since >= self.cam.get("parked_notify_after", 30) and v.hits >= 5:
            v.arrival_done = True
            # 停入前後要有畫面移動 (車輛駛入/牽入); 原本就停著、因光線變亮才偵測到的車不通知
            if any(v.first_seen - 180 <= t <= v.parked_since for t in self.motion_times):
                self.notify_arrival(frame, v)
            else:
                log.info("[%s] 車輛 #%d (%s) 停入前畫面無移動, 視為原本就停著, 不通知", self.name_, v.id, v.type_name)
        for i in near:
            self.parked_spots[i] = (cx, cy, now)
        if v.arrival_done and not near:
            self.parked_spots.append((cx, cy, now))

    def notify_arrival(self, frame, v):
        stamp = datetime.now()
        fname = self._snapshot(frame, [(v.box, v.type_name, (255, 160, 0))], stamp, "arrival")
        log.info("[%s] 🚗 車輛 #%d (%s) 停入 -> %s", self.name_, v.id, v.type_name, fname)
        self._send("\n".join([
            f"🚗 {self.label}：有{TYPE_ZH.get(v.type_name, '車輛')}停入",
            f"📷 鏡頭：{self.label}",
            f"🕒 時間：{stamp:%Y-%m-%d %H:%M:%S}",
        ]), fname)

    def mark_parked_spots(self, frame, now):
        """已停妥的車 (不論是否已通知) 都更新所在位置, 讓同一台車的重複追蹤不再通知."""
        h, w = frame.shape[:2]
        for v in self.vehicles.tracks:
            if v.parked_since is None or v.last_seen != now or not v.arrival_done:
                continue
            cx, cy = (v.box[0] + v.box[2]) / 2 / w, (v.box[1] + v.box[3]) / 2 / h
            for i, (x, y, _) in enumerate(self.parked_spots):
                if ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 < 0.08:
                    self.parked_spots[i] = (x, y, now)

    def annotate(self, frame, now):
        img = frame.copy()
        h, w = img.shape[:2]
        lw = max(1, w // 600)
        if self.zone_px is not None and self.active:
            cv2.polylines(img, [self.zone_px.astype(np.int32)], True, (0, 255, 255), lw + 1)
        if self.active:
            for z in self.ignore_px:
                cv2.polylines(img, [z.astype(np.int32)], True, (0, 0, 200), lw)
        dwell = self.cam.get("dwell_seconds", 10)
        for v in self.vehicles.tracks:
            if now - v.last_seen > self.vcfg["interval"] * 2.5:
                continue
            x1, y1, x2, y2 = map(int, v.box)
            color = (0, 255, 0) if v.plate else (255, 160, 0) if v.in_zone else (160, 160, 160)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
            parked = f"P {(now - v.parked_since) / 60:.0f}m" if v.parked_since else ""
            label = " ".join(x for x in (f"#{v.id}", v.plate or "", parked) if x)
            cv2.putText(img, label, (x1 + 5, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * lw, color, lw)
        for t in self.tracks:
            if t.last_seen != now:
                continue
            x1, y1, x2, y2 = map(int, t.box)
            inside = self.in_zone(t.box)
            color = (0, 255, 0) if t.result else (0, 165, 255) if inside else (160, 160, 160)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, lw + 1)
            still = min(now - t.still_since, dwell)
            label = (t.result and t.result[0]) or t.last_read or ("" if not inside else f"{still:.0f}/{dwell}s")
            cv2.putText(img, label, (max(0, x1 - 30), max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * lw, color, lw)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            LIVE[self.name_] = jpg.tobytes()

    def run(self):
        self.reader.start()
        period = 1.0 / self.cam.get("process_fps", 2)
        while True:
            start = time.time()
            frame = self.reader.frame()
            if frame is not None:  # 鏡頭斷線時是單色畫面 (上下有文字), 略過
                h = frame.shape[0]
                if frame[h // 10:h * 9 // 10:8, ::8].std() < 3:
                    frame = None
            if frame is not None:
                self.last_good = start
                if self.offline_alerted:
                    self.offline_alerted = False
                    log.info("[%s] 鏡頭恢復連線", self.name_)
                    self._send_text(f"✅ 鏡頭恢復連線：{self.label}")
                try:
                    self.process(frame, start)
                except Exception:  # noqa: BLE001
                    log.exception("[%s] 處理錯誤", self.name_)
            else:
                self.check_offline(start)
            time.sleep(max(0.05, period - (time.time() - start)))


# ---------------------------------------------------------------- Web (即時畫面 / 截圖)
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(DATA_DIR), **kw)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            imgs = ""
            for n, w in WORKERS.items():  # 依設定檔順序
                extra = f' <span data-status="{n}">{html_escape(w.status) if w else ""}</span>'
                if w is not None and w.home is not None:
                    extra += (f' <button onclick="if(confirm(\'確認 {n} 目前在原位? 將以目前畫面作為原位參考\'))'
                             f'fetch(\'/home/{n}\',{{method:\'POST\'}}).then(r=>r.text()).then(alert)">設為原位參考</button>'
                             f' <a href="/home/{n}/">參考畫面</a>')
                imgs += f'<h3>{html_escape(w.label if w else n)} <small>({n})</small>{extra}</h3><img src="/live/{n}.jpg" data-cam="{n}" style="max-width:100%">'

            html = (f"<html><head><meta charset='utf-8'><title>LPR</title></head><body style='font-family:sans-serif'>"
                    f"<h2>車牌辨識即時畫面</h2><p>黃框=電子圍籬 橘=圍籬內等待中 綠=已辨識 灰=圍籬外 紅=排除區(水溝蓋等) | "
                    f"<a href='/snapshots/'>歷史截圖</a></p>{imgs}"
                    "<script>setInterval(()=>document.querySelectorAll('img[data-cam]').forEach("
                    "i=>i.src='/live/'+i.dataset.cam+'.jpg?'+Date.now()),1000);"
                    "setInterval(()=>fetch('/status.json').then(r=>r.json()).then(s=>document.querySelectorAll("
                    "'[data-status]').forEach(e=>e.textContent=s[e.dataset.status]||'')),2000)</script></body></html>")
            return self._send(html.encode(), "text/html; charset=utf-8")
        if self.path == "/status.json":
            body = json.dumps({n: w.status for n, w in WORKERS.items()}, ensure_ascii=False)
            return self._send(body.encode(), "application/json; charset=utf-8")
        if self.path.startswith("/live/"):
            name = self.path[6:].split(".jpg")[0]
            if name in LIVE:
                return self._send(LIVE[name], "image/jpeg")
            return self.send_error(404)
        return super().do_GET()

    def do_POST(self):
        # 以目前畫面作為旋轉鏡頭的原位參考
        name = self.path[len("/home/"):].strip("/") if self.path.startswith("/home/") else ""
        w = WORKERS.get(name)
        if w is None or w.home is None or getattr(w, "last_frame", None) is None:
            return self.send_error(404)
        fname = f"{datetime.now():%Y%m%d_%H%M%S}.jpg"
        w.home.save_reference(w.last_frame, fname)
        log.info("[%s] 已儲存原位參考畫面 %s", name, fname)
        return self._send(f"已儲存原位參考畫面 {fname}".encode(), "text/plain; charset=utf-8")

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------- 進入點
def build_models(cfg):
    det = cfg["detection"]
    avail = ort.get_available_providers()
    prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
    log.info("ONNX Runtime providers: %s", prov)
    # 限制 CPU 執行緒, 避免吃滿整台主機
    threads = int(det.get("cpu_threads", 1))
    cv2.setNumThreads(1)

    def opts():
        o = ort.SessionOptions()
        o.intra_op_num_threads, o.inter_op_num_threads = threads, 1
        return o

    detector = DefaultDetector(det.get("plate_detector", "yolo-v9-t-384-license-plate-end2end"),
                               det["plate_conf"], prov, opts())
    ocr = PlateOCR(det.get("ocr_models", ["cct-xs-v2-global-model", "cct-s-v2-global-model"]),
                   device="cuda" if "CUDAExecutionProvider" in prov else "cpu", sess_options=opts())
    vehicle_models = {}
    model_dir = Path(os.environ.get("MODEL_DIR", "/models"))
    # 模型本身用所有鏡頭中最低的門檻, 各鏡頭再依自己的 vehicle_conf 篩選
    vconf = min([cfg.get("vehicle", {}).get("conf", 0.2)] + [c["vehicle_conf"] for c in cfg["cameras"] if "vehicle_conf" in c])

    def vehicle(name, pad):
        """車輛偵測模型 (同一模型/留白設定共用)."""
        key = (name, pad)
        if key not in vehicle_models:
            vehicle_models[key] = VehicleDetector(str(model_dir / name), vconf, providers=prov,
                                                  sess_options=opts(), pad=pad)
        return vehicle_models[key]

    return SimpleNamespace(detector=detector, ocr=ocr, decoder=Decoder(ocr.alphabet, ocr.pad_char), vehicle=vehicle)


VEHICLE_DEFAULTS = {
    "model": "yolov8n-416.onnx", "pad": 0.15, "conf": 0.2, "interval": 2.0, "min_area": 0.01,
    "zone_overlap": 0.3, "lost_seconds": 60, "relink_seconds": 600, "startup_grace": 60,
    "unidentified_after": 120, "unidentified_cooldown": 86400,
}
GALLERY_DEFAULTS = {"enabled": True, "interval": 300, "max_per_track": 6, "max_per_plate": 300, "save_others": True}


def worker_cfg(cfg, syn):
    return dict(cfg["detection"], snapshot_base_url=syn.get("snapshot_base_url"),
                vehicle=dict(VEHICLE_DEFAULTS, **(cfg.get("vehicle") or {})),
                gallery=dict(GALLERY_DEFAULTS, **(cfg.get("gallery") or {})))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/config/config.yaml")
    ap.add_argument("--test-image", help="對單張圖片測試車牌辨識後離開")
    ap.add_argument("--test-message", action="store_true", help="發送一則測試訊息到 Synology Chat 後離開")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug or os.environ.get("DEBUG") else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("urllib3", "open_image_models", "fast_alpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    syn = cfg["synology"]
    chat = SynologyChat(syn["webhook_url"], syn.get("verify_ssl", True))

    if args.test_message:
        ok = chat.send("✅ 車牌辨識系統測試訊息 (LPR test)")
        print("發送成功" if ok else "發送失敗")
        return

    models = build_models(cfg)
    whitelist = Whitelist(Path(args.config).with_name("whitelist.yaml"))

    if args.test_image:
        img = cv2.imread(args.test_image)
        cam = {"name": "test", "url": "", "zone": None}
        worker = CameraWorker(cam, worker_cfg(cfg, syn), models, chat, Cooldowns("/tmp/test-cooldowns.json"),
                              whitelist, Gallery("/tmp/test-gallery"))
        for *b, c, k in worker.vehicle_det.detect(img):
            print(f"vehicle {k} box={[int(v) for v in b]} conf={c:.2f}")
        for d in models.detector.predict(img):
            box = (d.bounding_box.x1, d.bounding_box.y1, d.bounding_box.x2, d.bounding_box.y2)
            probs, plate, conf = worker.read(img, box)
            entry, like = whitelist.best(probs, models.decoder) if probs is not None else (None, 0)
            wl = f" 白名單最接近={entry['plate']}({like:.2f})" if entry else ""
            print(f"plate box={box} det={d.confidence:.2f} -> {plate} conf={conf:.2f}{wl}")
        return

    det_cfg = worker_cfg(cfg, syn)
    cooldowns = Cooldowns(DATA_DIR / "cooldowns.json")
    gallery = Gallery(DATA_DIR / "gallery", cfg.get("gallery", {}).get("max_per_plate", 300))
    for cam in cfg["cameras"]:
        CameraWorker(cam, det_cfg, models, chat, cooldowns, whitelist, gallery).start()

    port = cfg.get("web", {}).get("port", 8090)
    log.info("Web 介面: http://0.0.0.0:%d/", port)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
