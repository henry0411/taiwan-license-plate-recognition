"""YOLOv8n (ONNX, COCO) 車輛偵測."""
import cv2
import numpy as np
import onnxruntime as ort

# COCO 類別
CLASS_NAMES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # 1 (腳踏車) 只在鏡頭設定 vehicle_classes 含 1 時偵測


class VehicleDetector:
    def __init__(self, model_path, conf=0.35, classes=(2, 3, 5, 7), providers=None, sess_options=None, pad=0.15):
        self.sess = ort.InferenceSession(model_path, sess_options=sess_options,
                                         providers=providers or ["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input, self.size = inp.name, inp.shape[2]
        self.conf, self.classes = conf, list(classes)
        self.pad = pad  # 四周留白比例: 被畫面切掉一部分的近距離車輛較容易偵測

    def detect(self, img, classes=None):
        """回傳 [(x1, y1, x2, y2, conf, class_id), ...] (原圖座標)."""
        h, w = img.shape[:2]
        r = self.size / (max(h, w) * (1 + 2 * self.pad))
        nh, nw = int(round(h * r)), int(round(w * r))
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.input: x})[0][0].T  # (8400, 84): cx, cy, w, h, 80 類分數
        cls_ids = np.array(classes if classes is not None else self.classes)
        scores = out[:, 4 + cls_ids]
        best = scores.argmax(1)
        conf = scores[np.arange(len(scores)), best]
        keep = conf >= self.conf
        if not keep.any():
            return []
        boxes, conf, cls = out[keep, :4], conf[keep], cls_ids[best[keep]]
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - left) / r
        xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - top) / r
        xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2 - left) / r
        xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2 - top) / r
        xyxy = np.clip(xyxy, 0, [w, h, w, h])
        # 不分類別的 NMS (同一台車被判成 car / truck 時只留一個)
        idx = cv2.dnn.NMSBoxes([[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in xyxy],
                               conf.tolist(), self.conf, 0.5)
        return [(*map(float, xyxy[i]), float(conf[i]), int(cls[i])) for i in np.array(idx).flatten()]
