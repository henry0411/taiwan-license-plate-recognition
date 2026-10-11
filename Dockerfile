# 台灣車牌辨識 (LPR) 服務
# 建置: docker compose build            (CPU)
#       GPU=1 docker compose build      (NVIDIA GPU, 需 nvidia-container-toolkit)
FROM python:3.12-slim
ARG GPU=0
ENV PYTHONUNBUFFERED=1
RUN if [ "$GPU" = "1" ]; then \
        pip install --no-cache-dir "fast-alpr[onnx-gpu]" ; \
    else \
        pip install --no-cache-dir "fast-alpr[onnx]" ; \
    fi \
    && pip install --no-cache-dir opencv-python-headless requests pyyaml numpy

# 預先下載車牌偵測 / OCR 模型, 避免執行時才下載
RUN python -c "from fast_alpr import ALPR; ALPR(detector_model='yolo-v9-t-384-license-plate-end2end', ocr_model='cct-xs-v2-global-model', ocr_device='cpu')"
RUN python -c "from fast_plate_ocr import LicensePlateRecognizer; LicensePlateRecognizer('cct-s-v2-global-model', device='cpu')"

# 車輛偵測 (YOLOv8n COCO, 由 ultralytics 匯出的 ONNX)
COPY models/ /models/
ENV MODEL_DIR=/models

WORKDIR /app
COPY app/ /app/
CMD ["python", "main.py", "--config", "/config/config.yaml"]
