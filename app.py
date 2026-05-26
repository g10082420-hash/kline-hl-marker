# -*- coding: utf-8 -*-
"""
K線頭部 / 底部自動標記 v11

最終邏輯：
1. 先找事件點
   - 白點 = 突破 5MA：昨 close < 5MA 且今 close > 5MA
   - 紅點 = 跌破 5MA：昨 close > 5MA 且今 close < 5MA

2. 假設今日為 T，從 T 往前回推
   - a = 第一個跌破 5MA 之日，含 T
   - b = 第一個突破 5MA 之日，含 T
   - 若 a 早於 b，L1 = a 與 b 之間 low 最小值，含 a、b
   - 若 a 晚於 b，H1 = a 與 b 之間 high 最大值，含 a、b
   - c = a 往前第一個跌破 5MA 之日，不含 a
   - d = b 往前第一個突破 5MA 之日，不含 b
   - L2 = b 與 c 之間 low 最小值，含 b、c
   - H2 = a 與 d 之間 high 最大值，含 a、d

3. 收盤基準點
   - 紅K：實體上邊
   - 綠K：實體下邊
"""

import io
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:
    streamlit_image_coordinates = None


st.set_page_config(page_title="K線頭部底部標記 v11", layout="wide")


# =========================
# 預設參數
# =========================
DEFAULTS = {
    "crop_top_ratio": 0.31,
    "crop_bottom_ratio": 0.66,
    "crop_left_ratio": 0.00,
    "crop_right_ratio": 0.95,
    "crop_padding_ratio": 0.00,
    "tolerance_px": 0,
}

# 紅K、綠K、橘色5MA偵測門檻
# 若券商截圖顏色不同，可在這裡微調
RED_K = dict(r_min=170, g_max=125, b_max=125, rg_gap=45)
GREEN_K = dict(g_min=115, r_max=135, b_max=135, gr_gap=28)
ORANGE_MA = dict(r_min=120, g_min=45, g_max=205, b_max=155, rg_gap=8, gb_gap=3)


# =========================
# 字型
# =========================
def get_font(size: int, prefer_chinese: bool = False):
    candidates = []

    if prefer_chinese:
        candidates += [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyhbd.ttc",
        ]

    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]

    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass

    return ImageFont.load_default()


# =========================
# 小工具
# =========================
def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def moving_median(arr, k=31):
    out = np.empty_like(arr, dtype=float)
    n = len(arr)
    half = k // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(arr[lo:hi])

    return out


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def draw_main_crop_box(img: Image.Image, x0: int, y0: int, x1: int, y1: int):
    out = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)
    draw.rectangle([x0, y0, x1, y1], outline=(255, 180, 0, 255), width=4)
    return out.convert("RGB")


# =========================
# 偵測 K 棒
# =========================
def find_row_segments(row_counts, max_gap=2):
    rows = np.where(row_counts > 0)[0]

    if len(rows) == 0:
        return []

    segments = []
    start = int(rows[0])
    prev = int(rows[0])

    for row in rows[1:]:
        row = int(row)

        if row - prev <= max_gap + 1:
            prev = row
            continue

        area = int(row_counts[start:prev + 1].sum())
        segments.append((start, prev, area))
        start = row
        prev = row

    area = int(row_counts[start:prev + 1].sum())
    segments.append((start, prev, area))

    return segments


def dominant_vertical_span(sub_mask, merge_gap=2, bridge_gap=8):
    row_counts = sub_mask.sum(axis=1)
    segments = find_row_segments(row_counts, max_gap=merge_gap)

    if not segments:
        return None

    primary = max(
        segments,
        key=lambda item: (item[2], item[1] - item[0] + 1),
    )
    top, bottom, _ = primary

    changed = True
    while changed:
        changed = False
        for seg_top, seg_bottom, _ in segments:
            if seg_bottom < top:
                gap = top - seg_bottom - 1
            elif seg_top > bottom:
                gap = seg_top - bottom - 1
            else:
                gap = 0

            if gap <= bridge_gap and (seg_top < top or seg_bottom > bottom):
                top = min(top, seg_top)
                bottom = max(bottom, seg_bottom)
                changed = True

    return top, bottom


def detect_candles(crop: np.ndarray, offset_x: int, offset_y: int):
    h, w = crop.shape[:2]

    r = crop[:, :, 0].astype(int)
    g = crop[:, :, 1].astype(int)
    b = crop[:, :, 2].astype(int)

    # 5MA 橘線本身也很容易符合「紅色」門檻。
    # 若不先扣掉，橘線會把多根 K 棒橫向串成一大段，最後被寬度上限濾掉。
    ma_like = (
        (r > ORANGE_MA["r_min"])
        & (g > ORANGE_MA["g_min"])
        & (g < ORANGE_MA["g_max"])
        & (b < ORANGE_MA["b_max"])
        & ((r - g) > ORANGE_MA["rg_gap"])
        & ((g - b) > ORANGE_MA["gb_gap"])
    )

    red = (
        (r > RED_K["r_min"])
        & (g < RED_K["g_max"])
        & (b < RED_K["b_max"])
        & ((r - g) > RED_K["rg_gap"])
        & (~ma_like)
    )

    green = (
        (g > GREEN_K["g_min"])
        & (r < GREEN_K["r_max"])
        & (b < GREEN_K["b_max"])
        & ((g - r) > GREEN_K["gr_gap"])
        & (~ma_like)
    )

    candle_mask = red | green

    # 以 x 欄位聚合，抓每根K棒可能範圍
    col_counts = candle_mask.sum(axis=0)
    active = col_counts >= 3

    runs = []
    max_gap = 1  # 避免把兩根很近的K棒包在一起
    i = 0

    while i < w:
        if not active[i]:
            i += 1
            continue

        start = i
        last = i
        gap = 0
        i += 1

        while i < w:
            if active[i]:
                last = i
                gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    break
            i += 1

        runs.append((start, last))

    candles = []

    for start, end in runs:
        width = end - start + 1

        # 太窄多半雜訊；太寬可能是價格標籤或文字
        if width < 3 or width > 42:
            continue

        sub_mask_all = candle_mask[:, start:end + 1]
        span = dominant_vertical_span(sub_mask_all)

        if span is None:
            continue

        span_top, span_bottom = span
        sub_mask = sub_mask_all[span_top:span_bottom + 1]
        ys, xs = np.where(sub_mask)

        if len(ys) < 18:
            continue

        y_min = int(ys.min()) + span_top
        y_max = int(ys.max()) + span_top
        height = y_max - y_min + 1

        if height < 10 or height > int(h * 0.95):
            continue

        red_count = int(red[span_top:span_bottom + 1, start:end + 1].sum())
        green_count = int(green[span_top:span_bottom + 1, start:end + 1].sum())

        color = "red" if red_count >= green_count else "green"
        color_mask = red[:, start:end + 1] if color == "red" else green[:, start:end + 1]

        # 避免同一個 x 欄位上方的圖例 / 標籤污染實體判斷。
        color_mask = color_mask.copy()
        color_mask[:span_top, :] = False
        color_mask[span_bottom + 1:, :] = False

        # 找K棒實體：同一列顏色像素多者較像實體，1~2像素較像影線
        row_counts = color_mask.sum(axis=1)
        max_row = int(row_counts.max())

        body_threshold = max(3, int(max_row * 0.52))
        body_rows = np.where(row_counts >= body_threshold)[0]

        if len(body_rows) == 0:
            body_rows = np.where(row_counts >= max(2, int(max_row * 0.35)))[0]

        if len(body_rows) == 0:
            body_top = y_min
            body_bottom = y_max
        else:
            body_top = int(body_rows.min())
            body_bottom = int(body_rows.max())

        # 使用者定義：
        # 紅K收盤基準點 = 實體上邊
        # 綠K收盤基準點 = 實體下邊
        close_y = body_top if color == "red" else body_bottom
        x_center = (start + end) // 2

        candles.append(
            {
                "x_crop": x_center,
                "x": x_center + offset_x,
                "x1": start + offset_x,
                "x2": end + offset_x,
                "width": width,
                "color": color,
                "y_high": y_min + offset_y,
                "y_low": y_max + offset_y,
                "close_y": close_y + offset_y,
                "body_top": body_top + offset_y,
                "body_bottom": body_bottom + offset_y,
            }
        )

    candles.sort(key=lambda c: c["x_crop"])

    # 只在框真的重疊 / 幾乎貼住時才合併，避免兩根K棒被包成一根
    merged = []
    for c in candles:
        if merged and c["x1"] <= merged[-1]["x2"] + 1:
            old_h = merged[-1]["y_low"] - merged[-1]["y_high"]
            new_h = c["y_low"] - c["y_high"]
            if new_h > old_h:
                merged[-1] = c
        else:
            merged.append(c)

    for i, c in enumerate(merged):
        c["index"] = i

    return merged


# =========================
# 偵測 5MA 橘線
# =========================
def detect_ma_y(crop: np.ndarray, offset_y: int):
    h, w = crop.shape[:2]

    r = crop[:, :, 0].astype(int)
    g = crop[:, :, 1].astype(int)
    b = crop[:, :, 2].astype(int)

    yy = np.arange(h)[:, None]

    orange = (
        (r > ORANGE_MA["r_min"])
        & (g > ORANGE_MA["g_min"])
        & (g < ORANGE_MA["g_max"])
        & (b < ORANGE_MA["b_max"])
        & ((r - g) > ORANGE_MA["rg_gap"])
        & ((g - b) > ORANGE_MA["gb_gap"])
    )

    # 避開主圖內上方文字與下方日期/勾選列
    orange = orange & (yy > int(h * 0.08)) & (yy < int(h * 0.90))

    xs = []
    ys = []

    for x in range(w):
        yvals = np.where(orange[:, x])[0]
        if len(yvals) > 0:
            xs.append(x)
            ys.append(float(np.median(yvals)))

    if len(xs) < 10:
        raise RuntimeError("找不到 5MA 橘線。請確認截圖有勾選 5MA，並調整主圖裁切範圍。")

    xs = np.array(xs)
    ys = np.array(ys)
    all_x = np.arange(w)

    raw = np.interp(all_x, xs, ys)
    smooth = moving_median(raw, 41)

    # 排除橘色文字 / 標籤雜訊
    keep = np.abs(ys - smooth[xs]) < 35

    if keep.sum() > 10:
        ma = np.interp(all_x, xs[keep], ys[keep])
        ma = moving_median(ma, 13)
    else:
        ma = smooth

    return ma + offset_y


# =========================
# 事件點：白點 / 紅點
# =========================
def build_event_points(candles, tolerance_px=0):
    """
    事件判定：

    白點 = 突破 5MA：
        昨日收盤基準點在 5MA 下方，
        今日收盤基準點在 5MA 上方。

    紅點 = 跌破 5MA：
        昨日收盤基準點在 5MA 上方，
        今日收盤基準點在 5MA 下方。

    收盤基準點：
        紅K = 實體上邊
        綠K = 實體下邊

    若某根K棒落在 tolerance_px 內，狀態視為 neutral，
    不會觸發突破 / 跌破。
    """

    up_events = []    # 白點：突破
    down_events = []  # 紅點：跌破

    # raw_status:
    # above   = 收盤基準點高於5MA
    # below   = 收盤基準點低於5MA
    # neutral = 貼線區，不直接觸發事件
    raw_status = []

    for c in candles:
        rel = c["rel"]

        # y座標越小，價格越高
        # rel < 0：收盤基準點在5MA上方
        # rel > 0：收盤基準點在5MA下方
        if rel < -tolerance_px:
            raw_status.append("above")
        elif rel > tolerance_px:
            raw_status.append("below")
        else:
            raw_status.append("neutral")

    for i in range(1, len(candles)):
        prev = raw_status[i - 1]
        curr = raw_status[i]

        # 突破：昨 close < 5MA，今 close > 5MA
        if prev == "below" and curr == "above":
            up_events.append(i)

        # 跌破：昨 close > 5MA，今 close < 5MA
        if prev == "above" and curr == "below":
            down_events.append(i)

    return up_events, down_events


# =========================
# 重點：依 T 回推 a / b / c / d 後產生 H-L
# =========================
def build_hl_from_events(candles, up_events, down_events):
    """
    假設今日為 T，也就是最後一根 K 棒。

    每輪都從目前 T 往前找：
    a = 第一個跌破 5MA 之日，含 T
    b = 第一個突破 5MA 之日，含 T

    若 a 早於 b，代表區間為跌破 -> 突破，取 a 與 b 之間的
    low 最小值，標 L。

    若 a 晚於 b，代表區間為突破 -> 跌破，取 a 與 b 之間的
    high 最大值，標 H。

    完成一段後，把 T 移到該段左側事件，繼續往左遞推，
    直到找不到成對的突破 / 跌破事件為止。

    區間含兩端；同高 / 同低取較右邊。
    """

    labels = []

    if not candles:
        return labels, labels

    current_idx = len(candles) - 1
    l_count = 0
    h_count = 0

    def append_label(label_type, name, event_1, event_2, source):
        left_idx, right_idx = sorted((event_1, event_2))
        segment = range(left_idx, right_idx + 1)

        if label_type == "L":
            # y_low 越大代表價格越低；同低取較右。
            idx = max(segment, key=lambda j: (candles[j]["y_low"], j))
        else:
            # y_high 越小代表價格越高；同高取較右。
            idx = min(segment, key=lambda j: (candles[j]["y_high"], -j))

        labels.append(
            {
                "idx": idx,
                "type": label_type,
                "name": name,
                "left_idx": left_idx,
                "right_idx": right_idx,
                "source": source,
                "status": "已計算",
                "provisional": False,
            }
        )

    while current_idx >= 0:
        a = latest_event_before_or_at(down_events, current_idx)
        b = latest_event_before_or_at(up_events, current_idx)

        if a is None or b is None:
            break

        if a < b:
            l_count += 1
            append_label("L", f"L{l_count}", a, b, "跌破→突破，區間取最低 low")
            current_idx = a
        elif a > b:
            h_count += 1
            append_label("H", f"H{h_count}", a, b, "突破→跌破，區間取最高 high")
            current_idx = b
        else:
            break

    # 畫圖用：由左到右畫
    labels_for_draw = sorted(labels, key=lambda x: (x["idx"], x["type"], x["name"]))

    # 表格用：由右到左看
    labels_for_table = sorted(labels, key=lambda x: x["right_idx"], reverse=True)

    return labels_for_draw, labels_for_table


def latest_event_before_or_at(events, idx):
    for event_idx in reversed(events):
        if event_idx <= idx:
            return event_idx

    return None


# =========================
# 日期 / 價格估算
# =========================
def estimate_date(index, n, start_date, end_date):
    if not start_date or not end_date or n <= 1:
        return ""

    try:
        est = start_date + (end_date - start_date) * (index / (n - 1))
        return est.strftime("%Y-%m-%d")
    except Exception:
        return ""


def estimate_price_from_y(y, chart_top_y, chart_bottom_y, price_top, price_bottom):
    if price_top is None or price_bottom is None:
        return None

    if chart_bottom_y == chart_top_y:
        return None

    ratio = (chart_bottom_y - y) / (chart_bottom_y - chart_top_y)
    price = price_bottom + ratio * (price_top - price_bottom)

    return float(price)


# =========================
# 畫事件點：白點 / 紅點
# =========================
def draw_event_points(draw, candles, up_events, down_events):
    # 白點：突破
    for idx in up_events:
        c = candles[idx]
        x = c["x"]
        y = c["close_y"]
        r = max(3, min(5, int(c["width"] * 0.55)))

        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=(255, 255, 255, 255),
            outline=(0, 0, 0, 255),
            width=1,
        )

    # 紅點：跌破
    for idx in down_events:
        c = candles[idx]
        x = c["x"]
        y = c["close_y"]
        r = max(3, min(5, int(c["width"] * 0.55)))

        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=(255, 0, 0, 255),
            outline=(255, 255, 255, 255),
            width=1,
        )


# =========================
# 畫 H / L
# =========================
def draw_labels(
    img: Image.Image,
    candles,
    labels_for_draw,
    up_events,
    down_events,
    crop_y0: int,
    crop_y1: int,
    display_mode: str = "HL",
    draw_box: bool = True,
    draw_events: bool = True,
    label_scale: float = 1.6,
):
    out = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)

    use_chinese = display_mode == "頭底"

    if draw_events:
        draw_event_points(draw, candles, up_events, down_events)

    plot_top = min(c["y_high"] for c in candles) if candles else crop_y0
    plot_bottom = max(c["y_low"] for c in candles) if candles else crop_y1

    for item in labels_for_draw:
        idx = item["idx"]
        typ = item["type"]
        c = candles[idx]

        if display_mode == "HL":
            text = "H" if typ == "H" else "L"
        else:
            text = "頭" if typ == "H" else "底"

        # 使用者原本要求：直徑不可超過K棒。
        # 但手機圖上常太小，這裡提供 label_scale；若要嚴格版，設 1.0。
        diam = max(7, int(c["width"] * label_scale))
        diam = min(diam, 22)
        radius = diam / 2

        font_size = max(8, int(diam * 0.9))
        font = get_font(font_size, prefer_chinese=use_chinese)

        x = c["x"]

        if typ == "H":
            above_y = c["y_high"] - diam - 4
            if above_y < plot_top:
                y = min(c["y_high"] + 4, crop_y1 - diam - 4)
            else:
                y = max(crop_y0 + 2, above_y)
            text_color = (255, 0, 0, 255)
            box_color = (255, 0, 0, 255)
        else:
            below_y = c["y_low"] + 4
            if below_y + diam > plot_bottom:
                y = max(crop_y0 + 2, c["y_low"] - diam - 4)
            else:
                y = min(crop_y1 - diam - 4, below_y)
            text_color = (0, 190, 0, 255)
            box_color = (0, 190, 0, 255)

        if draw_box:
            pad = 3
            draw.rectangle(
                [c["x1"] - pad, c["y_high"] - pad, c["x2"] + pad, c["y_low"] + pad],
                outline=box_color,
                width=2,
            )

        # 白色圓形外框
        draw.ellipse(
            [x - radius, y, x + radius, y + diam],
            outline=(255, 255, 255, 255),
            width=max(1, int(diam * 0.16)),
            fill=(0, 0, 0, 0),
        )

        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        draw.text(
            (x - tw / 2, y + diam / 2 - th / 2 - 1),
            text,
            font=font,
            fill=text_color,
        )

    return out.convert("RGB")


# =========================
# 手動標註
# =========================
MANUAL_TARGETS = ["突破點", "跌破點", "H", "L"]


def nearest_candle_by_x(candles, x: float, max_x_dist: int):
    if not candles:
        return None

    c = min(candles, key=lambda item: abs(item["x"] - x))
    if abs(c["x"] - x) <= max_x_dist:
        return c

    return None


def snap_manual_mark_to_candle(mark, candle):
    target = mark.get("target")
    mark["idx"] = int(candle["index"])
    mark["snapped"] = True
    mark["x"] = float(candle["x"])

    if target == "突破點" or target == "跌破點":
        mark["y"] = float(candle["close_y"])
    elif target == "H":
        mark["y"] = float(candle["y_high"])
    elif target == "L":
        mark["y"] = float(candle["y_low"])

    return mark


def build_manual_mark(target: str, x: float, y: float, candles, max_x_dist: int):
    c = nearest_candle_by_x(candles, x, max_x_dist)

    mark = {
        "target": target,
        "raw_x": float(x),
        "raw_y": float(y),
        "x": float(x),
        "y": float(y),
        "idx": None,
        "snapped": False,
    }

    if c is None:
        return mark

    return snap_manual_mark_to_candle(mark, c)


def move_manual_mark(mark, candles, offset: int):
    if not candles or mark.get("idx") is None:
        return mark

    candle_by_idx = {int(c["index"]): c for c in candles}
    current_idx = int(mark["idx"])
    next_idx = clamp(current_idx + offset, 0, len(candles) - 1)
    next_candle = candle_by_idx.get(next_idx)

    if next_candle is None:
        return mark

    mark["adjust_count"] = int(mark.get("adjust_count", 0)) + abs(next_idx - current_idx)

    return snap_manual_mark_to_candle(mark, next_candle)


def draw_manual_annotations(
    img: Image.Image,
    marks,
    candles,
    crop_y0: int,
    crop_y1: int,
    draw_box: bool = True,
    label_scale: float = 1.6,
):
    out = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)
    candle_by_idx = {c["index"]: c for c in candles}

    for mark in marks:
        target = mark.get("target")
        idx = mark.get("idx")
        c = candle_by_idx.get(idx)
        x = float(mark.get("x", mark.get("raw_x", 0)))
        y = float(mark.get("y", mark.get("raw_y", 0)))

        if target in ("突破點", "跌破點"):
            width = c["width"] if c else 12
            radius = max(4, min(7, int(width * 0.65)))

            if target == "突破點":
                fill = (255, 255, 255, 255)
                outline = (0, 0, 0, 255)
            else:
                fill = (255, 0, 0, 255)
                outline = (255, 255, 255, 255)

            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=fill,
                outline=outline,
                width=2,
            )
            continue

        if target not in ("H", "L"):
            continue

        width = c["width"] if c else 12
        diam = max(11, int(width * label_scale))
        diam = min(diam, 28)
        radius = diam / 2
        font_size = max(9, int(diam * 0.82))
        font = get_font(font_size, prefer_chinese=False)

        if target == "H":
            label_y = max(crop_y0 + 2, y - diam - 4)
            text_color = (255, 0, 0, 255)
            box_color = (255, 0, 0, 255)
        else:
            label_y = min(crop_y1 - diam - 4, y + 4)
            text_color = (0, 210, 0, 255)
            box_color = (0, 210, 0, 255)

        if draw_box and c:
            pad = 3
            draw.rectangle(
                [c["x1"] - pad, c["y_high"] - pad, c["x2"] + pad, c["y_low"] + pad],
                outline=box_color,
                width=2,
            )

        draw.ellipse(
            [x - radius, label_y, x + radius, label_y + diam],
            outline=(255, 255, 255, 255),
            width=max(1, int(diam * 0.16)),
            fill=(0, 0, 0, 0),
        )

        bbox = draw.textbbox((0, 0), target, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        draw.text(
            (x - tw / 2, label_y + diam / 2 - th / 2 - 1),
            target,
            font=font,
            fill=text_color,
        )

    return out.convert("RGB")


def manual_rows(marks):
    rows = []
    for i, mark in enumerate(marks, start=1):
        rows.append(
            {
                "順序": i,
                "目標": mark.get("target"),
                "K棒序號": "" if mark.get("idx") is None else int(mark["idx"]) + 1,
                "吸附": "是" if mark.get("snapped") else "否",
                "左右修正": int(mark.get("adjust_count", 0)),
                "X": round(float(mark.get("x", 0)), 1),
                "Y": round(float(mark.get("y", 0)), 1),
            }
        )

    return rows


def detect_candles_for_box(img: Image.Image, crop_box):
    x0, y0, x1, y1 = crop_box
    crop_img = img.crop((x0, y0, x1, y1))
    crop = np.array(crop_img)
    candles = detect_candles(crop, x0, y0)
    return crop_img, candles


def render_manual_annotation_tab(
    img: Image.Image,
    crop_box,
    upload_key: str,
    draw_box: bool,
    label_scale: float,
    auto_img: Image.Image = None,
):
    x0, y0, x1, y1 = crop_box
    safe_key = "".join(ch if ch.isalnum() else "_" for ch in upload_key)[-80:]
    marks_key = f"manual_marks_{safe_key}"
    last_click_key = f"manual_last_click_{safe_key}"

    if marks_key not in st.session_state:
        st.session_state[marks_key] = []
    if last_click_key not in st.session_state:
        st.session_state[last_click_key] = None

    marks = st.session_state[marks_key]

    st.subheader("手動標註")

    base_options = ["完全原圖"]
    if auto_img is not None:
        base_options.insert(0, "自動標記圖")

    base_choice = st.radio(
        "手動底稿",
        base_options,
        horizontal=True,
        key=f"manual_base_{safe_key}",
    )
    use_auto_base = base_choice == "自動標記圖" and auto_img is not None
    base_img = auto_img if use_auto_base else img
    base_key = "auto" if use_auto_base else "original"

    c1, c2 = st.columns([2, 1])
    with c1:
        target = st.radio("目前目標", MANUAL_TARGETS, horizontal=True, key=f"manual_target_{safe_key}")
    with c2:
        snap_dist = st.slider(
            "K棒吸附距離",
            5,
            80,
            35,
            1,
            key=f"manual_snap_{safe_key}",
        )

    try:
        crop_img, candles = detect_candles_for_box(img, crop_box)
    except Exception as e:
        crop_img = img.crop(crop_box)
        candles = []
        st.warning(f"K棒吸附暫時不可用：{e}")

    active_mark = marks[-1] if marks else None
    can_adjust = bool(active_mark and active_mark.get("idx") is not None and candles)
    active_idx = int(active_mark["idx"]) if can_adjust else None

    if can_adjust:
        st.caption(f"目前修正：第 {len(marks)} 筆 / {active_mark.get('target')} / 第 {active_idx + 1} 根K棒")
    elif marks:
        st.caption(f"目前修正：第 {len(marks)} 筆 / 未吸附到K棒")
    else:
        st.caption("目前修正：尚無手動標註")

    m1, m2 = st.columns(2)
    if m1.button(
        "← 往左一根",
        disabled=not can_adjust or active_idx <= 0,
        key=f"manual_left_{safe_key}",
        use_container_width=True,
    ):
        marks[-1] = move_manual_mark(marks[-1], candles, -1)
        st.session_state[marks_key] = marks
        st.rerun()

    if m2.button(
        "往右一根 →",
        disabled=not can_adjust or active_idx >= len(candles) - 1,
        key=f"manual_right_{safe_key}",
        use_container_width=True,
    ):
        marks[-1] = move_manual_mark(marks[-1], candles, 1)
        st.session_state[marks_key] = marks
        st.rerun()

    b1, b2 = st.columns(2)
    if b1.button("復原上一筆", disabled=not marks, key=f"manual_undo_{safe_key}", use_container_width=True):
        marks.pop()
        st.session_state[marks_key] = marks
        st.rerun()

    if b2.button("清空手動標註", disabled=not marks, key=f"manual_clear_{safe_key}", use_container_width=True):
        st.session_state[marks_key] = []
        st.rerun()

    full_annotated = draw_manual_annotations(
        img=base_img,
        marks=marks,
        candles=candles,
        crop_y0=y0,
        crop_y1=y1,
        draw_box=draw_box,
        label_scale=label_scale,
    )
    click_img = full_annotated.crop(crop_box)

    if streamlit_image_coordinates is None:
        st.error("手動標註需要 streamlit-image-coordinates。請先執行：pip install streamlit-image-coordinates")
        st.image(click_img, use_container_width=True)
    else:
        click = streamlit_image_coordinates(
            click_img,
            use_column_width="always",
            key=f"manual_click_{safe_key}_{base_key}",
            cursor="crosshair",
        )

        if click and click.get("unix_time") != st.session_state[last_click_key]:
            if click.get("width") and click.get("height"):
                raw_x = x0 + click["x"] * (click_img.width / click["width"])
                raw_y = y0 + click["y"] * (click_img.height / click["height"])
                marks.append(build_manual_mark(target, raw_x, raw_y, candles, snap_dist))
                st.session_state[marks_key] = marks
                st.session_state[last_click_key] = click.get("unix_time")
                st.rerun()

    st.download_button(
        "下載手動標註圖 PNG",
        data=pil_to_png_bytes(full_annotated),
        file_name=f"manual_marked_{base_key}.png",
        mime="image/png",
        disabled=not marks,
        key=f"manual_download_{safe_key}",
    )

    if marks:
        st.dataframe(manual_rows(marks), use_container_width=True)
    else:
        st.info("尚未建立手動標註。")


# =========================
# 主處理函式
# =========================
def annotate_kline_image(
    img: Image.Image,
    crop_top_ratio=0.31,
    crop_bottom_ratio=0.66,
    crop_left_ratio=0.00,
    crop_right_ratio=0.95,
    crop_padding_ratio=0.00,
    tolerance_px=0,
    display_mode="HL",
    draw_box=True,
    draw_events=True,
    label_scale=1.6,
    price_top=None,
    price_bottom=None,
    start_date=None,
    end_date=None,
):
    img = img.convert("RGB")
    W, H = img.size

    pad_x = int(W * crop_padding_ratio)

    x0 = int(W * crop_left_ratio) + pad_x
    x1 = int(W * crop_right_ratio) - pad_x
    y0 = int(H * crop_top_ratio)
    y1 = int(H * crop_bottom_ratio)

    x0 = clamp(x0, 0, W - 1)
    x1 = clamp(x1, 1, W)
    y0 = clamp(y0, 0, H - 1)
    y1 = clamp(y1, 1, H)

    if y1 <= y0 or x1 <= x0:
        raise RuntimeError("裁切範圍不正確，請調整主圖上下左右邊界。")

    crop_img = img.crop((x0, y0, x1, y1))
    crop = np.array(crop_img)

    candles = detect_candles(crop, x0, y0)

    if len(candles) < 5:
        raise RuntimeError("偵測到的 K 棒太少，請調整主圖裁切範圍。")

    ma_y = detect_ma_y(crop, y0)
    crop_w = crop.shape[1]

    for i, c in enumerate(candles):
        xc = clamp(c["x_crop"], 0, crop_w - 1)

        c["index"] = i
        c["ma_y"] = float(ma_y[xc])

        # rel < 0：收盤基準點在5MA上方
        # rel > 0：收盤基準點在5MA下方
        c["rel"] = float(c["close_y"] - c["ma_y"])

    up_events, down_events = build_event_points(candles, tolerance_px=tolerance_px)

    labels_for_draw, labels_for_table = build_hl_from_events(
        candles,
        up_events=up_events,
        down_events=down_events,
    )

    result = draw_labels(
        img=img,
        candles=candles,
        labels_for_draw=labels_for_draw,
        up_events=up_events,
        down_events=down_events,
        crop_y0=y0,
        crop_y1=y1,
        display_mode=display_mode,
        draw_box=draw_box,
        draw_events=draw_events,
        label_scale=label_scale,
    )

    rows = []

    for item in labels_for_table:
        idx = item["idx"]
        typ = item["type"]
        c = candles[idx]

        price_y = c["y_high"] if typ == "H" else c["y_low"]
        price = estimate_price_from_y(price_y, y0, y1, price_top, price_bottom)
        est_date = estimate_date(idx, len(candles), start_date, end_date)

        rows.append(
            {
                "類型": "頭部" if typ == "H" else "底部",
                "標記": "H" if typ == "H" else "L",
                "代號": item["name"],
                "狀態": item["status"],
                "依據": item["source"],
                "標記K棒序號": idx + 1,
                "右側事件序號": item["right_idx"] + 1,
                "左側事件序號": item["left_idx"] + 1,
                "區間": f'{item["left_idx"] + 1} ~ {item["right_idx"] + 1}',
                "估算日期": est_date,
                "估算價格": "" if price is None else round(price, 2),
                "K棒顏色": "紅K" if c["color"] == "red" else "綠K",
            }
        )

    info = {
        "candles": len(candles),
        "up_events": len(up_events),
        "down_events": len(down_events),
        "labels": len(labels_for_draw),
        "confirmed_labels": sum(1 for item in labels_for_draw if not item.get("provisional")),
        "provisional_labels": sum(1 for item in labels_for_draw if item.get("provisional")),
        "crop_box": (x0, y0, x1, y1),
        "crop_img": crop_img,
        "up_event_indices": [i + 1 for i in up_events],
        "down_event_indices": [i + 1 for i in down_events],
    }

    return result, rows, info


# =========================
# Streamlit UI
# =========================
st.title("K線頭部 / 底部 自動標記 v11")
st.caption("v11：改善上方圖例污染 K 棒高低點；H/L 採 T 往左遞推完整標記。")

with st.sidebar:
    st.header("標示設定")

    display_mode = st.radio(
        "標示模式",
        ["HL", "頭底"],
        index=0,
        help="HL = H/L；頭底 = 頭/底",
    )

    draw_box = st.checkbox("圈出被判定的H/L K棒", value=True)
    draw_events = st.checkbox("顯示事件點：白點突破、紅點跌破", value=True)

    label_scale = st.slider(
        "H/L 標記大小倍率",
        1.0,
        3.0,
        1.6,
        0.1,
        help="若要嚴格遵守直徑不超過K棒，請調成 1.0。",
    )

    st.divider()
    st.header("主圖裁切")

    crop_top_ratio = st.slider("主圖上緣", 0.00, 0.85, DEFAULTS["crop_top_ratio"], 0.01)
    crop_bottom_ratio = st.slider("主圖下緣", 0.20, 1.00, DEFAULTS["crop_bottom_ratio"], 0.01)
    crop_left_ratio = st.slider("主圖左邊界", 0.00, 0.40, DEFAULTS["crop_left_ratio"], 0.01)
    crop_right_ratio = st.slider("主圖右邊界", 0.50, 1.00, DEFAULTS["crop_right_ratio"], 0.01)

    crop_padding_ratio = st.slider(
        "裁切保護邊距",
        0.00,
        0.08,
        DEFAULTS["crop_padding_ratio"],
        0.005,
        help="預設0。若右側價格軸或浮動按鈕被誤抓，可稍微調大；若第一根/最後一根K棒被切掉，請調回0。",
    )

    st.divider()
    st.header("判定設定")

    tolerance_px = st.slider(
        "貼線容許誤差",
        0,
        20,
        DEFAULTS["tolerance_px"],
        1,
        help="建議先用0，代表嚴格判定。若5MA與收盤基準點太貼，可改成1~3。數字太大容易漏抓或誤判。",
    )

    st.header("日期 / 價格估算")
    st.caption("目前不是 OCR。若要明細顯示日期與價格，請手動輸入價格軸與日期範圍。")

    use_est = st.checkbox("啟用估算日期 / 價格", value=False)

    price_top = None
    price_bottom = None
    start_date = None
    end_date = None

    if use_est:
        price_top = st.number_input("主圖上緣價格", value=800.0, step=1.0)
        price_bottom = st.number_input("主圖下緣價格", value=650.0, step=1.0)
        start_date = st.date_input("左側第一根K棒約略日期")
        end_date = st.date_input("右側最後一根K棒約略日期")


uploaded = st.file_uploader("上傳 K 線截圖", type=["png", "jpg", "jpeg", "webp"])


if uploaded:
    img = Image.open(uploaded).convert("RGB")
    W, H = img.size

    pad_x = int(W * crop_padding_ratio)

    x0 = int(W * crop_left_ratio) + pad_x
    x1 = int(W * crop_right_ratio) - pad_x
    y0 = int(H * crop_top_ratio)
    y1 = int(H * crop_bottom_ratio)

    x0 = clamp(x0, 0, W - 1)
    x1 = clamp(x1, 1, W)
    y0 = clamp(y0, 0, H - 1)
    y1 = clamp(y1, 1, H)

    preview = draw_main_crop_box(img, x0, y0, x1, y1)

    tab1, tab_manual, tab2, tab3, tab4 = st.tabs(["結果", "手動標註", "主圖裁切檢查", "事件明細", "原圖"])
    auto_result = None

    try:
        result, rows, info = annotate_kline_image(
            img=img,
            crop_top_ratio=crop_top_ratio,
            crop_bottom_ratio=crop_bottom_ratio,
            crop_left_ratio=crop_left_ratio,
            crop_right_ratio=crop_right_ratio,
            crop_padding_ratio=crop_padding_ratio,
            tolerance_px=tolerance_px,
            display_mode=display_mode,
            draw_box=draw_box,
            draw_events=draw_events,
            label_scale=label_scale,
            price_top=price_top,
            price_bottom=price_bottom,
            start_date=start_date,
            end_date=end_date,
        )
        auto_result = result

        with tab1:
            st.subheader("標記結果")
            st.image(result, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("K棒數", info["candles"])
            c2.metric("白點突破數", info["up_events"])
            c3.metric("紅點跌破數", info["down_events"])
            c4.metric("H/L標記數", info["labels"])
            st.caption("H/L 依最後一根 T 往左遞推，逐段完成所有成對的突破 / 跌破區間。")

            st.download_button(
                "下載標記圖 PNG",
                data=pil_to_png_bytes(result),
                file_name=f"marked_{display_mode}_v11.png",
                mime="image/png",
            )

            if rows:
                st.subheader("標記明細（由右往左排序）")
                st.dataframe(rows, use_container_width=True)
            else:
                st.warning("目前缺少可回推的突破或跌破事件，因此沒有 H/L 標記。")

        with tab2:
            st.subheader("主圖裁切檢查")
            st.caption("橘框應該只包住主K線圖，不要包到成交量、KD、MACD。")
            st.image(preview, use_container_width=True)

            st.subheader("實際裁切區")
            st.image(info["crop_img"], use_container_width=True)

        with tab3:
            st.subheader("事件點序號")
            st.write("白點突破事件序號：", info["up_event_indices"])
            st.write("紅點跌破事件序號：", info["down_event_indices"])

            st.caption("規則：每輪從T往前找a=第一個跌破、b=第一個突破；跌破→突破取L，突破→跌破取H；完成後把T移到該段左側事件繼續往左遞推。")

            if rows:
                st.subheader("H/L 區間明細")
                st.dataframe(rows, use_container_width=True)

        with tab4:
            st.subheader("原圖")
            st.image(img, use_container_width=True)

    except Exception as e:
        with tab1:
            st.error(f"處理失敗：{e}")

        with tab2:
            st.subheader("主圖裁切檢查")
            st.image(preview, use_container_width=True)

        with tab4:
            st.subheader("原圖")
            st.image(img, use_container_width=True)

    with tab_manual:
        upload_key = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}:{W}x{H}"
        render_manual_annotation_tab(
            img=img,
            crop_box=(x0, y0, x1, y1),
            upload_key=upload_key,
            draw_box=draw_box,
            label_scale=label_scale,
            auto_img=auto_result,
        )

else:
    st.info("請先上傳一張 K 線截圖。")
