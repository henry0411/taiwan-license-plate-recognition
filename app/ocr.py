"""車牌 OCR: 多模型 + 多種裁切取平均, 輸出每個字元位置的機率矩陣."""
import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer
from fast_plate_ocr.inference.plate_recognizer import _load_image_from_source, preprocess_image

# 裁切外擴比例 (x, y): 車牌框偵測有誤差, 多種裁切平均較穩定
PADS = [(0.05, 0.10), (0.0, 0.0), (0.12, 0.2)]


class PlateOCR:
    def __init__(self, models, device="cpu", providers=None, sess_options=None):
        self.recs = [LicensePlateRecognizer(m, device=device, providers=providers, sess_options=sess_options)
                     for m in models]
        cfg = self.recs[0].config
        for r in self.recs[1:]:
            if (r.config.alphabet, r.config.max_plate_slots) != (cfg.alphabet, cfg.max_plate_slots):
                raise ValueError("OCR 模型的字元集/長度不一致, 無法合併")
        self.alphabet, self.pad_char, self.slots = cfg.alphabet, cfg.pad_char, cfg.max_plate_slots

    def _crops(self, frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        for px, py in PADS:
            dx, dy = (x2 - x1) * px, (y2 - y1) * py
            c = frame[int(max(0, y1 - dy)):int(min(h, y2 + dy)), int(max(0, x1 - dx)):int(min(w, x2 + dx))]
            if c.size:
                yield c

    def probs(self, frame, box):
        """回傳 (slots, vocab) 平均機率, 或 None."""
        crops = list(self._crops(frame, box))
        if not crops:
            return None
        out = []
        for r in self.recs:
            mode = r.config.image_color_mode
            conv = cv2.COLOR_BGR2GRAY if mode == "grayscale" else cv2.COLOR_BGR2RGB
            batch = [cv2.cvtColor(c, conv) for c in crops]
            x = preprocess_image(_load_image_from_source(batch, r.config))
            y = r.model.run([r.plate_output_name], {"input": x})[0]
            out.append(y.reshape(len(crops), self.slots, -1))
        return np.concatenate(out).mean(axis=0)
