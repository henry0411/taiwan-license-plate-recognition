# 台灣車牌辨識系統（僅用 CPU，住宅場景）

**繁體中文** | [English](README.md)

這是一套住家用的車牌辨識與車輛識別服務。它跑在一台 4 核心 vCPU、4 GB 記憶體、沒有 GPU 的虛擬機上，處理一支戶外旋轉鏡頭和一支車庫內鏡頭，結果發送到 Synology Chat。

📄 **論文：** 中文 [PDF](paper/zh/paper.pdf) · [Markdown](paper/zh/paper.md) | English [PDF](paper/en/paper.pdf) · [Markdown](paper/en/paper.md)

## 重點成果（取自論文，皆為真實部署資料）

| 項目 | 結果 | 樣本 |
|---|---|---|
| 機車外觀辨識（DINOv2） | 報出車名時正確率 **92.3%**；之後的停入報出車名 **14/14** 全對（擴增參考圖前後皆然），擴增讓 top-1 從 89.3% 提高到 **96.4%** | 44 次停入（26 次報出車名）；之後的 28 次停入（14 次報出車名，其餘通知「有機車停入」） |
| 多畫面融合（5 張畫面） | 車牌解碼準確率 **91.8%**，同一組抽樣下單張 76.2%、平手比信心的多數決 87.1%；3 張畫面為 89.5% | 13 個車牌（3 張時 25 個），僅限可辨識車牌，只評估解碼（不含確認規則） |
| 似然式白名單比對 | 非白名單車牌誤認 **0/56**（上限 6.4%），白名單中有一位數之差的車號時也是；接受率 83.6%（逐字比對 77.2%） | 171 次可辨識車牌的單張測試，來自 56 個不同車牌 |
| 車框內第二次偵測車牌 | **59.6%** 的可辨識車牌**只**有這一步找得到 | 171 個可辨識車牌 |
| 停妥遲滯 | 重播中重複「停妥」事件減少 **44%**，17 條可通知軌跡全數保留；線上紀錄也減少（觀察性比較） | 1,500 張車庫畫面、25 天紀錄 |

車牌結果幾乎都來自戶外白天畫面，所有標註由同一位標註者完成（AI 助理，看不到系統輸出）。限制請見論文第 6 節。

## 運作方式

```
鏡頭 (MJPEG) ─► 畫面品質過濾 ─► 車輛偵測 (YOLOv8n) ─► 車輛追蹤 + 停妥遲滯
                           └──► 車牌偵測 (YOLOv9-t)：整張畫面 + 每個車框內各一次
                                └► OCR：2 個模型 × 3 種裁切，機率平均
                                   └► 台灣車牌格式約束解碼
                                      └► 跨畫面機率融合 ─► 白名單似然比對 ─► 發送通知
停妥但讀不到車牌 ─► 停入前後邊緣相似度（原本就停著嗎？）─► DINOv2 外觀比對 ─► 「機車 X 停入」
```

| 模組 | 檔案 |
|---|---|
| 鏡頭處理、車牌追蹤、融合、通知、網頁介面 | `app/main.py` |
| OCR（多模型、多裁切） | `app/ocr.py` |
| 台灣車牌格式約束解碼、白名單似然 | `app/tw_plate.py` |
| 車輛偵測（YOLOv8n ONNX） | `app/vehicle.py` |
| 車輛追蹤、停妥遲滯、邊緣相似度、外觀樣本 | `app/vehicles.py` |
| 機車外觀辨識（DINOv2） | `app/appearance.py` |
| 旋轉鏡頭原位判斷 | `app/home.py` |
| Synology Chat webhook | `app/notify.py` |

## 快速開始

> **本 repo 不附模型檔。** YOLOv8 權重採 AGPL-3.0 授權，所以不在這裡散布，請自行產生：

1. **車輛偵測模型：** 用 [Ultralytics](https://github.com/ultralytics/ultralytics) 匯出 YOLOv8n 的 ONNX：`yolo export model=yolov8n.pt format=onnx imgsz=640`，存成 `models/yolov8n.onnx`。範例設定中兩支啟用的鏡頭都使用這個 640 像素模型（`vehicle_pad: 0.0`）。全域預設的 `yolov8n-416.onnx`（`imgsz=416`）只有在鏡頭沒有指定 `vehicle_model` 時才需要。
2. **外觀辨識模型（選用）：** DINOv2-small 的 ONNX 檔，第一個輸出必須是 token 序列 `(1, 1+patches, 384)`。可以使用 Hugging Face 上的 `onnx-community/dinov2-small`，存成 `data/models/dinov2-small.onnx`。
3. 車牌偵測與 OCR 模型會在 `docker compose build` 時由 [fast-alpr](https://github.com/ankandrew/fast-alpr) 自動下載。

```bash
# 先編輯 config/config.yaml（webhook_url、鏡頭、區域）和 config/whitelist.yaml
docker compose build
docker compose up -d
# 網頁介面：http://<主機>:8091/
docker compose run --rm lpr python main.py --config /config/config.yaml --test-message   # 發送測試訊息
```

`config/config.yaml` 中要修改的項目：

- `synology.webhook_url`：Synology Chat 的傳入 webhook 網址，請勿外流。
- `cameras[].url`：MJPEG 串流網址。`camera-host`、`lpr-host`、`go2rtc-host` 都只是佔位名稱。
- `config/whitelist.yaml`：填入自己的車牌，修改後會自動重新載入。
- 機車參考圖：放在 `data/appearance/<名稱>/*.jpg`，每台約 10 張，白天和夜間都要有，修改後會自動重新載入。

## 重現實驗

`experiments/` 是論文使用的實驗腳本，在服務的映像檔（`lpr:cpu`）內執行：

| 腳本 | 論文章節 |
|---|---|
| `plate_extract.py` | 預先萃取車牌裁切圖，以及各模型、各裁切的 OCR 機率 |
| `plate_exp.py` | §5.3～5.6 OCR 消融、信心門檻、白名單 |
| `twopass_deployed.py` | §5.2 以部署的車輛偵測設定重算兩階段偵測 |
| `fusion_table.py`、`plate_stats_extra.py` | §5.5 多畫面融合與以車牌為單位的 bootstrap 區間 |
| `tracking_exp.py` | §5.7 停妥遲滯重播 |
| `appearance_exp.py` | §5.9 外觀特徵比較與參考圖擴增 |

`experiments/results/` 是各表格背後的彙總結果。原始影像與標註**不公開**，因為其中有第三方的車牌和人物。`tools/` 內有圖表產生器、隱私遮罩工具和 PDF 建置腳本（先執行 `npm install marked@12`，再執行 `CHROME=/path/to/chrome node tools/build_pdf.js paper/zh/paper.md paper/zh/paper.pdf`）。

## 分支

- `main`：目前的程式碼、論文與實驗。
- `history`：部署過程的 17 個去敏版本，時間從 2026-10-11 到 2026-11-14，由有日期的備份重建。

## 隱私

- 程式碼與 git 歷史中的金鑰（webhook token、NAS 主機名稱）、內部網路位址、真實白名單和提到車型的註解都已移除。
- 圖片只保留裁切後的局部畫面，車牌、人臉、品牌與車型標誌以及背景雜物都已馬賽克或模糊（車身外形仍可見）。文中的車牌一律寫成 `ABC-12XX`。
- 請只在自己擁有或已獲授權監看的場所使用，並遵守當地的隱私法規。

## 授權

本 repo 的程式碼採 [MIT](LICENSE) 授權。第三方模型各依其授權：YOLOv8（AGPL-3.0）、fast-alpr / fast-plate-ocr（MIT）、DINOv2（Apache-2.0）。
