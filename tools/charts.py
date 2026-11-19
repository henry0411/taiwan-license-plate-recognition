"""Static SVG charts for the paper (pure Python, no plotting library).

Palette: reference categorical slots 1-3 (blue #2a78d6, orange #eb6834, aqua #1baf7a) on a light surface;
the deployed configuration is the only saturated bar, alternatives are neutral gray.
Usage: python3 charts.py <exp_dir> <out_dir> <lang zh|en>
"""
import json
import sys
from pathlib import Path

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b7b0"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"
FONT = "'Noto Sans CJK TC','Noto Sans TC','Noto Sans',sans-serif"

T = {
    "zh": {
        "ocr_title": "單張畫面車牌辨識準確率（可辨識車牌 n=171）",
        "fusion_title": "多畫面融合：準確率 vs 畫面數 k（以車牌為單位平均）",
        "fusion_x": "每個車牌使用的畫面數 k", "single": "單張", "vote": "多數決（平手比信心）", "mean": "機率平均＋信心門檻（部署）",
        "track_title": "停妥後位移門檻 vs 「停妥」事件數（車庫重播）",
        "track_x": "停妥後位移門檻（車框寬度的倍數）", "track_y": "停妥事件數", "base": "未設（0.1）",
        "app_title": "機車外觀辨識：44 次真實停入的 top-1 準確率",
        "live_title": "車庫每日「停妥」記錄數（線上紀錄）", "before": "修正前", "after": "修正後（11/06 起）",
        "acc": "準確率",
        "ocr_labels": {
            "xs model, 1 crop, argmax": "xs・1 裁切・無格式約束",
            "s model, 1 crop, argmax": "s・1 裁切・無格式約束",
            "xs model, 1 crop, TW-format": "xs・1 裁切・台灣格式",
            "s model, 1 crop, TW-format": "s・1 裁切・台灣格式",
            "xs+s, 1 crop, TW-format": "xs+s・1 裁切・台灣格式",
            "xs, 3 crops, TW-format": "xs・3 裁切・台灣格式",
            "s, 3 crops, TW-format": "s・3 裁切・台灣格式",
            "xs+s, 3 crops, argmax": "xs+s・3 裁切・無格式約束",
            "xs+s, 3 crops, TW-format (deployed)": "xs+s・3 裁切・台灣格式（部署）",
        },
        "app_labels": {
            "dino_gray_cls+mean (deployed)": "DINOv2 灰階 CLS + patch 平均（部署）",
            "dino_gray_cls": "DINOv2 灰階 CLS",
            "dino_gray_patchmean": "DINOv2 灰階 patch 平均",
            "dino_rgb_cls+mean": "DINOv2 彩色 CLS + patch 平均",
            "hog": "HOG", "raw_pixels_32": "原始像素 32×32", "hsv_histogram": "HSV 顏色直方圖",
        },
    },
    "en": {
        "ocr_title": "Single-frame plate accuracy (readable plates, n=171)",
        "fusion_title": "Multi-frame fusion: accuracy vs number of frames k (plate-averaged)",
        "fusion_x": "frames per plate k", "single": "single frame", "vote": "majority vote (conf. tie-break)",
        "mean": "prob. average + gate (deployed)",
        "track_title": "Parked-state threshold vs number of 'parked' events (garage replay)",
        "track_x": "parked-state threshold (fraction of box width)", "track_y": "parked events", "base": "none (0.1)",
        "app_title": "Motorcycle identification: top-1 accuracy on 44 real arrivals",
        "live_title": "Daily 'parked' log entries, garage camera (production logs)", "before": "before fix",
        "after": "after fix (from Nov 6)", "acc": "accuracy",
        "ocr_labels": {
            "xs model, 1 crop, argmax": "xs, 1 crop, unconstrained",
            "s model, 1 crop, argmax": "s, 1 crop, unconstrained",
            "xs model, 1 crop, TW-format": "xs, 1 crop, TW format",
            "s model, 1 crop, TW-format": "s, 1 crop, TW format",
            "xs+s, 1 crop, TW-format": "xs+s, 1 crop, TW format",
            "xs, 3 crops, TW-format": "xs, 3 crops, TW format",
            "s, 3 crops, TW-format": "s, 3 crops, TW format",
            "xs+s, 3 crops, argmax": "xs+s, 3 crops, unconstrained",
            "xs+s, 3 crops, TW-format (deployed)": "xs+s, 3 crops, TW format (deployed)",
        },
        "app_labels": {
            "dino_gray_cls+mean (deployed)": "DINOv2 gray CLS + patch mean (deployed)",
            "dino_gray_cls": "DINOv2 gray CLS", "dino_gray_patchmean": "DINOv2 gray patch mean",
            "dino_rgb_cls+mean": "DINOv2 RGB CLS + patch mean", "hog": "HOG", "raw_pixels_32": "raw pixels 32x32",
            "hsv_histogram": "HSV color histogram",
        },
    },
}


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="{FONT}" role="img" aria-label="{title}">'
            f'<rect width="{w}" height="{h}" fill="{SURF}"/>'
            f'<text x="16" y="28" font-size="17" font-weight="600" fill="{INK}">{title}</text>{body}</svg>')


def hbar(rows, title, highlight, maxv=1.0, fmt=lambda v, n, d: f"{v:.1%}  ({n}/{d})"):
    """rows: [(label, value, n, d)]"""
    lw, bw, rh = 330, 360, 30
    h = 56 + rh * len(rows) + 30
    w = lw + bw + 150
    b = []
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        x = lw + bw * t / maxv
        b.append(f'<line x1="{x:.1f}" y1="48" x2="{x:.1f}" y2="{h - 26}" stroke="{GRID}" stroke-width="1"/>'
                 f'<text x="{x:.1f}" y="{h - 10}" font-size="12" fill="{INK2}" text-anchor="middle">{t:.0%}</text>')
    for k, (label, v, n, d) in enumerate(rows):
        y = 56 + k * rh
        col = BLUE if label == highlight else GRAY
        wpx = max(2, bw * v / maxv)
        weight = "600" if label == highlight else "400"
        b.append(f'<text x="{lw - 10}" y="{y + 15}" font-size="13" fill="{INK}" text-anchor="end" '
                 f'font-weight="{weight}">{label}</text>'
                 f'<rect x="{lw}" y="{y + 3}" width="{wpx:.1f}" height="{rh - 10}" rx="4" fill="{col}"/>'
                 f'<text x="{lw + wpx + 8:.1f}" y="{y + 16}" font-size="12" fill="{INK2}">{fmt(v, n, d)}</text>')
    return svg(w, h, "".join(b), title)


def line_chart(xs, series, title, xlabel, ylim=(0.6, 1.0), ylabel_fmt="{:.0%}", w=720, h=380, xpos=None):
    """series: [(label, color, [y...])]; xpos: numeric x positions (default: evenly spaced categories)"""
    L, R, TOP, B = 70, 230, 50, 60
    pw, ph = w - L - R, h - TOP - B
    xpos = xpos or list(range(len(xs)))
    sx = lambda i: L + pw * (xpos[i] - xpos[0]) / (xpos[-1] - xpos[0])  # noqa: E731
    sy = lambda v: TOP + ph * (1 - (v - ylim[0]) / (ylim[1] - ylim[0]))  # noqa: E731
    b = []
    steps = 5
    for k in range(steps):
        v = ylim[0] + (ylim[1] - ylim[0]) * k / (steps - 1)
        b.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{L + pw}" y2="{sy(v):.1f}" stroke="{GRID}"/>'
                 f'<text x="{L - 8}" y="{sy(v) + 4:.1f}" font-size="12" fill="{INK2}" text-anchor="end">'
                 f'{ylabel_fmt.format(v)}</text>')
    for i, x in enumerate(xs):
        b.append(f'<text x="{sx(i):.1f}" y="{TOP + ph + 20}" font-size="12" fill="{INK2}" text-anchor="middle">{x}</text>')
    b.append(f'<text x="{L + pw / 2}" y="{h - 14}" font-size="13" fill="{INK2}" text-anchor="middle">{xlabel}</text>')
    for label, col, ys in series:
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(ys))
        b.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        for i, v in enumerate(ys):
            b.append(f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="4.5" fill="{col}" stroke="{SURF}" stroke-width="2"/>')
        b.append(f'<text x="{sx(len(ys) - 1) + 10:.1f}" y="{sy(ys[-1]) + 4:.1f}" font-size="12" fill="{INK}">'
                 f'{label} {ylabel_fmt.format(ys[-1])}</text>')
    return svg(w, h, "".join(b), title)


def main(exp, out, lang):
    exp, out, t = Path(exp), Path(out), T[lang]
    out.mkdir(parents=True, exist_ok=True)
    pr = json.loads((exp / "plate_results.json").read_text())
    rows = [(t["ocr_labels"][k], v["readable"][0] / v["readable"][1], *v["readable"])
            for k, v in pr["single_frame"].items()]
    (out / "chart_ocr.svg").write_text(hbar(rows, t["ocr_title"], t["ocr_labels"]["xs+s, 3 crops, TW-format (deployed)"]))

    fu = json.loads((exp / "fusion_table.json").read_text())
    ks = ["1", "2", "3", "5"]
    get = lambda key: [fu[k][key] for k in ks]  # noqa: E731
    (out / "chart_fusion.svg").write_text(line_chart(
        ks, [(t["single"], GRAY, get("single")), (t["vote"], ORANGE, get("vote")),
             (t["mean"], BLUE, get("prob_avg_gated"))], t["fusion_title"], t["fusion_x"], xpos=[1, 2, 3, 5], w=780))

    ar = json.loads((exp / "appearance_results.json").read_text())["E1"]
    order = ["dino_gray_cls+mean (deployed)", "dino_gray_cls", "dino_rgb_cls+mean", "dino_gray_patchmean", "hog",
             "raw_pixels_32", "hsv_histogram"]
    rows = [(t["app_labels"][m], ar[m]["top1_correct"] / ar[m]["n"], ar[m]["top1_correct"], ar[m]["n"]) for m in order]
    (out / "chart_appearance.svg").write_text(hbar(rows, t["app_title"], t["app_labels"][order[0]]))

    tr = exp / "tracking_results.json"
    if tr.exists():
        sw = json.loads(tr.read_text())["sweep"]
        keys = ["None", "0.15", "0.2", "0.25", "0.3", "0.4", "0.5", "0.75", "1.0"]
        vals = [sw[k]["parked_events"] for k in keys]
        top = -(-max(vals) // 50) * 50
        labels = ["0.1" if k == "None" else ("" if k in ("0.15", "0.25") else k) for k in keys]
        xpos = [0.1 if k == "None" else float(k) for k in keys]
        top = 120
        (out / "chart_tracking.svg").write_text(line_chart(
            labels, [(t["track_y"], BLUE, vals)], t["track_title"], t["track_x"], ylim=(0, top),
            ylabel_fmt="{:.0f}", w=760, xpos=xpos))

    live = json.loads((exp / "live_stats.json").read_text())
    parked = live.get("garage_parked_log_entries") or live["parked"]
    days = [d for d in live["days"] if d < "2026-11-18"]
    vals = [parked.get(d, 0) for d in days]
    w, h, L, B, TOP = 760, 340, 60, 60, 70
    pw, ph = w - L - 20, h - TOP - B
    top = -(-max(vals) // 200) * 200
    bw = pw / len(days)
    b = [f'<rect x="{L}" y="40" width="12" height="12" rx="2" fill="{GRAY}"/>'
         f'<text x="{L + 18}" y="51" font-size="12" fill="{INK2}">{t["before"]}</text>'
         f'<rect x="{L + 120}" y="40" width="12" height="12" rx="2" fill="{BLUE}"/>'
         f'<text x="{L + 138}" y="51" font-size="12" fill="{INK2}">{t["after"]}</text>']
    for k in range(5):
        v = top * k / 4
        y = TOP + ph * (1 - v / top)
        b.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" stroke="{GRID}"/>'
                 f'<text x="{L - 8}" y="{y + 4:.1f}" font-size="12" fill="{INK2}" text-anchor="end">{v:.0f}</text>')
    for i, (d, v) in enumerate(zip(days, vals)):
        x = L + i * bw
        hh = ph * v / top
        col = BLUE if d >= "2026-11-06" else GRAY
        b.append(f'<rect x="{x + 1:.1f}" y="{TOP + ph - hh:.1f}" width="{bw - 2:.1f}" height="{hh:.1f}" rx="3" fill="{col}">'
                 f'<title>{d}: {v}</title></rect>')
        if i % 3 == 0:
            b.append(f'<text x="{x + bw / 2:.1f}" y="{TOP + ph + 18}" font-size="11" fill="{INK2}" '
                     f'text-anchor="middle">{d[5:]}</text>')
    (out / "chart_live.svg").write_text(svg(w, h, "".join(b), t["live_title"]))
    print("charts written to", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
