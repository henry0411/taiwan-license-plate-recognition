# Taiwan License Plate Recognition (CPU-only, residential)

[繁體中文](README.zh-TW.md) | **English**

A license plate recognition and vehicle identification service for a home. It runs on a 4-vCPU, 4 GB virtual machine with no GPU and handles an outdoor pan-tilt camera and an indoor garage camera. Results go to Synology Chat.

📄 **Paper:** English [PDF](paper/en/paper.pdf) · [Markdown](paper/en/paper.md) | 中文 [PDF](paper/zh/paper.pdf) · [Markdown](paper/zh/paper.md)

## Highlights (from the paper, real deployment data)

| What | Result | Sample |
|---|---|---|
| Motorcycle appearance ID (DINOv2) | **92.3%** correct whenever it names a vehicle; on later arrivals **14/14** named correctly (with or without augmentation), and reference augmentation lifts top-1 from 89.3% to **96.4%** | 44 arrivals (26 named); 28 later arrivals (14 named, the rest reported as "a motorcycle arrived") |
| Multi-frame fusion (5 frames) | **91.8%** plate decoding accuracy vs 76.2% for a single frame and 87.1% for a confidence-tie-break vote on the same draws; 3 frames: 89.5% | 13 plates (25 for 3 frames), readable plates only, decoding only (no confirmation rule) |
| Likelihood whitelist matching | **0/56** non-whitelisted plates accepted (upper bound 6.4%), also with a one-digit look-alike on the list; acceptance 83.6% (exact match 77.2%) | 171 single-frame tests of readable plates from 56 distinct plates |
| Second plate-detection pass inside vehicle boxes | **59.6%** of readable plates are found **only** by this pass | 171 readable plates |
| Parked-state hysteresis | **44%** fewer duplicate "parked" events in replay, all 17 notifiable tracks kept; fewer in production too (observational) | 1,500 garage frames; 25 days of logs |

The plate results are almost entirely from the outdoor camera in daylight, and all labels come from a single annotator (an AI assistant, blind to system output). See Section 6 of the paper.

## How it works

```
camera (MJPEG) ─► frame quality filter ─► vehicle detection (YOLOv8n) ─► vehicle tracking + parked hysteresis
                                     └──► plate detection (YOLOv9-t) on the full frame + inside each vehicle box
                                          └► OCR: 2 models × 3 crops, probabilities averaged
                                             └► Taiwan format-constrained decoding
                                                └► cross-frame probability fusion ─► whitelist likelihood match ─► notification
parked vehicle without a plate ─► before/after edge similarity (was it already there?) ─► DINOv2 appearance match ─► "motorcycle X arrived"
```

| Module | File |
|---|---|
| Camera workers, plate tracking, fusion, notifications, web UI | `app/main.py` |
| OCR (multi-model, multi-crop) | `app/ocr.py` |
| Taiwan format-constrained decoding, whitelist likelihood | `app/tw_plate.py` |
| Vehicle detector (YOLOv8n ONNX) | `app/vehicle.py` |
| Vehicle tracking, parked hysteresis, edge similarity, gallery | `app/vehicles.py` |
| Motorcycle / bicycle appearance identification (DINOv2) | `app/appearance.py` |
| Pan-tilt camera home-position check | `app/home.py` |
| Synology Chat webhook | `app/notify.py` |

## Quick start

> **Model files are not included.** The YOLOv8 weights are AGPL-3.0, so they are not redistributed here. Create them yourself:

1. **Vehicle detector:** export YOLOv8n to ONNX with [Ultralytics](https://github.com/ultralytics/ultralytics): `yolo export model=yolov8n.pt format=onnx imgsz=640`, saved as `models/yolov8n.onnx`. Both active cameras in the example config use this 640-px model with `vehicle_pad: 0.0`. The global default `yolov8n-416.onnx` (`imgsz=416`) is only needed for cameras that do not set `vehicle_model`.
2. **Appearance model (optional):** a DINOv2-small ONNX export whose first output is the token sequence `(1, 1+patches, 384)`. One option is the `onnx-community/dinov2-small` export on Hugging Face. Save it as `data/models/dinov2-small.onnx`.
3. The plate detector and OCR models are downloaded automatically by [fast-alpr](https://github.com/ankandrew/fast-alpr) during `docker compose build`.

```bash
# first edit config/config.yaml (webhook_url, cameras, zones) and config/whitelist.yaml
docker compose build
docker compose up -d
# web UI: http://<host>:8091/
docker compose run --rm lpr python main.py --config /config/config.yaml --test-message   # send a test message
```

Edit `config/config.yaml`:

- `synology.webhook_url`: your Synology Chat incoming-webhook URL (keep it secret).
- `cameras[].url`: MJPEG stream URLs. `camera-host`, `lpr-host` and `go2rtc-host` are placeholders.
- `config/whitelist.yaml`: your own plates. It is reloaded automatically. Vehicles without a plate (e.g. a bicycle) can be registered by appearance only: an entry with `appearance: <folder name>` and no `plate`.
- Motorcycle / bicycle references: `data/appearance/<name>/*.jpg`, about 10 crops per vehicle covering day and night. They are reloaded automatically. Set the vehicle type of non-motorcycle folders in `appearance.types` (e.g. `{my_bicycle: bicycle}`).
- Bicycles (COCO class 1) are detected only on cameras whose `vehicle_classes` include 1. Per-type thresholds: `vehicle_min_area_by_type` and `vehicle_conf_by_type`. Bicycles never adopt plates read nearby.

## Reproducing the experiments

`experiments/` contains the scripts used in the paper. They run inside the service image (`lpr:cpu`):

| Script | Paper section |
|---|---|
| `plate_extract.py` | caches plate crops and per-model / per-crop OCR probabilities |
| `plate_exp.py` | §5.3–5.6 OCR ablation, confidence gate, whitelist |
| `twopass_deployed.py` | §5.2 two-pass detection with the deployed vehicle detector |
| `fusion_table.py`, `plate_stats_extra.py` | §5.5 multi-frame fusion and plate-level bootstrap intervals |
| `tracking_exp.py` | §5.7 parked-state hysteresis replay |
| `appearance_exp.py` | §5.9 appearance features and reference augmentation |

`experiments/results/` holds the aggregated results behind every table. The raw frames and labels are **not** published, because they contain third-party plates and people. `tools/` has the chart generator, the privacy-masking helper and the PDF build script (`npm install marked@12`, then `CHROME=/path/to/chrome node tools/build_pdf.js paper/en/paper.md paper/en/paper.pdf`).

## Branches

- `main`: current code, paper and experiments.
- `history`: sanitized snapshots of the deployment from 2026-10-11 onward (17 reconstructed from dated backups up to 2026-11-14, then one per update).

## Privacy

- Secrets (webhook token, NAS host name), LAN addresses, the real whitelist and comments naming vehicle models have been removed from the code and the git history.
- Figures are cropped to local regions; plates, faces, brand and model badges and background clutter are pixelated or blurred (vehicle bodies remain visible). Plates in the text are written as `ABC-12XX`.
- Only use this on property you own or are authorized to monitor, and follow local privacy law.

## License

[MIT](LICENSE) for the code in this repository. Third-party models keep their own licenses: YOLOv8 (AGPL-3.0), fast-alpr / fast-plate-ocr (MIT), DINOv2 (Apache-2.0).
