"""旋轉鏡頭的原位偵測: 與原位參考畫面做 ORB 特徵比對.

鏡頭在原位 (或僅輕微偏移) 時回傳參考畫面 -> 目前畫面的 homography, 用來校正電子圍籬;
鏡頭轉走時回傳 None.
參考畫面放在 <dir>/*.jpg, 可放多張 (例如白天 / 夜間紅外線各一張), 符合任一張即可.

自動學習: 確認在原位時, 每隔 learn_interval 秒把目前畫面記為「近期原位畫面」(僅存記憶體),
之後也拿來比對. 停車、日夜光線、下雨等逐漸變化都能跟上, 不必每種情況都手動存參考畫面.
近期原位畫面都記錄其相對於參考畫面的 homography, 比對結果一律換算回參考畫面座標再檢查偏移量,
因此不會一次次累積漂移.
"""
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("home")
# 比對用的固定尺寸: 參考畫面與目前畫面都縮放到此尺寸, 因此不受解析度 / 長寬比變更影響
SIZE = (640, 400)
CLAHE = cv2.createCLAHE(3.0, (8, 8))


def _prep(img):
    gray = cv2.cvtColor(cv2.resize(img, SIZE), cv2.COLOR_BGR2GRAY)
    gray = CLAHE.apply(gray)  # 對比增強, 讓白天/夜間畫面較容易比對
    mask = np.zeros_like(gray)
    gh = gray.shape[0]
    mask[gh // 10:gh * 9 // 10] = 255  # 排除上下的時間戳記文字
    return gray, mask


class HomeCheck:
    def __init__(self, ref_dir, min_inliers=20, max_shift=0.08, learn_interval=300, learn_inliers=40, learn_keep=12):
        self.dir = Path(ref_dir)
        self.min_inliers, self.max_shift = min_inliers, max_shift
        # 近期原位畫面 [(時間, kp, des, 參考 -> 該畫面的 homography)]
        self.learned, self.last_learn = [], 0.0
        self.learn_interval, self.learn_inliers, self.learn_keep = learn_interval, learn_inliers, learn_keep
        self.orb = cv2.ORB_create(5000, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.refs, self.mtime = [], None
        self.lock = threading.Lock()
        self.cache = None  # (縮圖, 時間, 結果): 畫面沒什麼變化時沿用上次比對結果, 省 CPU

    def _features(self, img):
        gray, mask = _prep(img)
        return self.orb.detectAndCompute(gray, mask)

    def _reload(self):
        mtime = self.dir.stat().st_mtime if self.dir.exists() else None
        if mtime == self.mtime:
            return
        self.mtime, self.refs = mtime, []
        for p in sorted(self.dir.glob("*.jpg")) if mtime else []:
            img = cv2.imread(str(p))
            if img is None:
                continue
            kp, des = self._features(img)
            if des is not None and len(kp) >= self.min_inliers:
                self.refs.append((p.name, kp, des))
        log.info("[%s] 原位參考畫面: %d 張", self.dir.name, len(self.refs))

    @property
    def has_reference(self):
        with self.lock:
            self._reload()
            return bool(self.refs)

    def save_reference(self, frame, name):
        self.dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.dir / name), frame)

    def check(self, frame, now, max_diff=8.0, max_age=30.0):
        """同 match(), 但畫面與上次完整比對時差異很小 (鏡頭沒轉動) 就沿用上次結果."""
        thumb = cv2.cvtColor(cv2.resize(frame, (64, 40), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.cache is not None and now - self.cache[1] < max_age and self.mtime is not None:
            if float(np.abs(thumb - self.cache[0]).mean()) < max_diff and self.dir.stat().st_mtime == self.mtime:
                return self.cache[2]
        result = self.match(frame, now)
        self.cache = (thumb, now, result)
        return result

    def match(self, frame, now=None):
        """回傳 (參考 -> 目前畫面 的 3x3 homography (正規化座標 0~1), 內點數) 或 (None, 最佳內點數)."""
        with self.lock:
            self._reload()
            refs = [(rkp, rdes, None) for _, rkp, rdes in self.refs]
            refs += [(rkp, rdes, base) for _, rkp, rdes, base in self.learned]
        if not self.refs:
            return None, 0
        kp, des = self._features(frame)
        if des is None or len(kp) < self.min_inliers:
            return None, 0
        best_h, best_h_n, best_n = None, 0, 0
        norm = np.float32(SIZE)
        corners = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]).reshape(-1, 1, 2)
        for rkp, rdes, base in refs:
            pairs = self.matcher.knnMatch(rdes, des, k=2)
            good = [m for m, *n in pairs if n and m.distance < 0.8 * n[0].distance]
            if len(good) < self.min_inliers:
                best_n = max(best_n, len(good))
                continue
            src = np.float32([rkp[m.queryIdx].pt for m in good]) / norm
            dst = np.float32([kp[m.trainIdx].pt for m in good]) / norm
            H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0 / SIZE[0])
            n = int(inl.sum()) if inl is not None else 0
            best_n = max(best_n, n)
            if H is None or n < self.min_inliers:
                continue
            if base is not None:  # 近期原位畫面: 換算回參考畫面座標
                H = H @ base
            # 四個角落的平均位移 (相對於參考畫面) 需夠小, 才算「在原位」(只容許輕微偏移)
            shift = np.linalg.norm(cv2.perspectiveTransform(corners, H) - corners, axis=2).mean()
            if shift <= self.max_shift and n > best_h_n:
                best_h, best_h_n = H, n
        if best_h is None:
            return None, best_n
        if now is not None and best_h_n >= self.learn_inliers and now - self.last_learn >= self.learn_interval:
            with self.lock:
                self.last_learn = now
                self.learned = (self.learned + [(now, kp, des, best_h)])[-self.learn_keep:]
        return best_h, best_h_n
