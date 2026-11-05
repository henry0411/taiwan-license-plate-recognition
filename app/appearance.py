"""機車外觀辨識 (DINOv2-small): 讀不到車牌的自家機車, 以外觀比對是哪一台.

參考圖放在 data/appearance/<名稱>/*.jpg (整台車的裁切圖, 白天與夜間紅外線畫面都要有),
新增/刪除檔案後自動重新載入, 不需重啟.
"""
import logging
import threading
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

log = logging.getLogger("appearance")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def crop(frame, box, pad=0.05):
    """車框裁切 (四周略留白)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    return frame[y1:y2, x1:x2]


class Appearance:
    def __init__(self, model_path, ref_dir, providers=None, sess_options=None):
        self.sess = ort.InferenceSession(str(model_path), sess_options=sess_options,
                                         providers=providers or ["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.ref_dir = Path(ref_dir)
        self.refs = {}       # 名稱 -> (N, D) 特徵
        self.stamp = None    # 參考圖檔案清單與修改時間 (變動時重新載入)
        self.cache = {}      # 檔案路徑 -> (mtime, 特徵)
        self.lock = threading.Lock()

    def embed(self, img):
        """轉灰階 (白天彩色 / 夜間紅外線畫面一致), 補成正方形後縮放到 224, 取 CLS + patch 平均並正規化."""
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        s = max(h, w)
        sq = np.full((s, s), 114, np.uint8)
        sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = g
        x = cv2.resize(sq, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = (np.repeat(x[:, :, None], 3, 2) - MEAN) / STD
        out = self.sess.run(None, {self.input: x.transpose(2, 0, 1)[None]})[0][0]   # (1 + patches, D)
        f = np.concatenate([out[0], out[1:].mean(0)])
        return f / (np.linalg.norm(f) + 1e-9)

    def _reload(self):
        files = sorted(p for p in self.ref_dir.glob("*/*.jpg") if not p.parent.name.startswith("_"))
        stamp = tuple((str(p), p.stat().st_mtime) for p in files)
        if stamp == self.stamp:
            return
        refs = {}
        for path, mtime in stamp:
            hit = self.cache.get(path)
            if hit is None or hit[0] != mtime:
                img = cv2.imread(path)
                if img is None:
                    continue
                hit = self.cache[path] = (mtime, self.embed(img))
            refs.setdefault(Path(path).parent.name, []).append(hit[1])
        self.refs = {k: np.stack(v) for k, v in refs.items()}
        self.stamp = stamp
        log.info("外觀參考圖: %s", ", ".join(f"{k} {len(v)} 張" for k, v in self.refs.items()) or "無")

    def identify(self, img):
        """回傳 [(名稱, 相似度), ...] 由高到低 (相似度 = 與該車參考圖的最大餘弦相似度)."""
        with self.lock:
            self._reload()
            if not self.refs:
                return []
            f = self.embed(img)
            return sorted(((k, float((v @ f).max())) for k, v in self.refs.items()), key=lambda x: -x[1])
