# -*- coding: utf-8 -*-
"""
K線頭部 / 底部自動標記網頁版
修正版：依「從右邊往左邊回推」規則配對突破 / 跌破 5MA。

使用方式：
1. pip install -r requirements.txt
2. streamlit run app.py
3. iPhone Safari 開啟網址，上傳截圖
"""

import io
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


st.set_page_config(page_title="K線頭部底部標記", layout="wide")


# =========================
# 預設參數
# =========================
DEFAULTS = {
    "crop_top_ratio": 0.30,
    "crop_bottom_ratio": 0.65,
    "crop_left_ratio": 0.00,
    "crop_right_ratio": 0.89,
    "tolerance_px": 6,
}

# 紅K、綠K、橘色5MA 偵測門檻
# 不同券商截圖顏色略有差異，必要時再微調
RED_K = dict(r_min=170, g_max=120, b_max=120, rg_gap=45)
GREEN_K = dict(g_min=115, r_max=130, b_max=130, gr_gap=30)
ORANGE_MA = dict(r_min=120, g_min=45, g_max=200, b_max=150, rg_gap=10, gb_gap=5)


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
# 偵測K棒
# =========================
def detect_candles(crop: np.ndarray, offset_x: int, offset_y: int):
    h, w = crop.shape[:2]

    r = crop[:, :, 0].astype(int)
    g = crop[:, :, 1].astype(int)
    b = crop[:, :, 2].astype(int)

    red = (
        (r > RED_K["r_min"])
        & (g < RED_K["g_max"])
        & (b < RED_K["b_max"])
        & ((r - g) > RED_K["rg_gap"])
    )

    green = (
        (g > GREEN_K["g_min"])
        & (r < GREEN_K["r_max"])
        & (b < GREEN_K["b_max"])
        & ((g - r) > GREEN_K["gr_gap"])
    )

    candle_mask = red | green

    # 以 x 欄位聚合，抓出每根K棒可能所在區段
    col_counts = candle_mask.sum(axis=0)
    active = col_counts >= 3

    runs = []
    max_gap = 5
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
        if width < 3 or width > 45:
            continue

        sub_mask = candle_mask[:, start:end + 1]
        ys, xs = np.where(sub_mask)

        if len(ys) < 20:
            continue

        y_min = int(ys.min())
        y_max = int(ys.max())
        height = y_max - y_min + 1

        if height < 12 or height > int(h * 0.95):
            continue

        red_count = int(red[:, start:end + 1].sum())
        green_count = int(green[:, start:end + 1].sum())

        color = "red" if red_count >= green_count else "green"
        color_mask = red[:, start:end + 1] if color == "red" else green[:, start:end + 1]

        # 實體判斷：同一列顏色像素多的區段較可能是K棒實體
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
        # 紅K：收盤價 = 實體上緣
        # 綠K：收盤價 = 實體下緣
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

    # 合併距離太近的偵測結果，避免同一根K棒被拆成兩根
    merged = []
    for c in candles:
        if merged and c["x_crop"] - merged[-1]["x_crop"] < 11:
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
# 偵測5MA橘線
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

    # 避開主圖內上方說明文字與下方日期/勾選列
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

    # 排除被文字或標籤誤抓到的橘色雜訊
    keep = np.abs(ys - smooth[xs]) < 35

    if keep.sum() > 10:
        ma = np.interp(all_x, xs[keep], ys[keep])
        ma = moving_median(ma, 13)
    else:
        ma = smooth

    return ma + offset_y


# =========================
# 重點：從右往左配對事件
# =========================
def build_labels_right_to_left(candles, tolerance_px=6):
    """
    使用者最終規則：

    L：
    從最右邊往左找「收盤突破5MA，且昨收未突破」的第1根、第2根，
    在兩者區間內找最低點，標 L。

    H：
    從最右邊往左找「收盤跌破5MA，且昨收未跌破」的第1根、第2根，
    在兩者區間內找最高點，標 H。

    區間均含兩個基準點。
    """

    down_breaks = []  # 收盤跌破5MA：昨收未跌破，今收跌破
    up_breaks = []    # 收盤突破5MA：昨收未突破，今收突破

    # 先由左到右找出所有事件位置
    for i in range(1, len(candles)):
        prev_rel = candles[i - 1]["rel"]
        curr_rel = candles[i]["rel"]

        # y 越大代表價格越低
        # rel > 0：收盤在 5MA 下方
        # rel < 0：收盤在 5MA 上方

        # 跌破5MA：昨天未跌破，今天跌破
        if prev_rel <= tolerance_px and curr_rel > tolerance_px:
            down_breaks.append(i)

        # 突破5MA：昨天未突破，今天突破
        if prev_rel >= -tolerance_px and curr_rel < -tolerance_px:
            up_breaks.append(i)

    labels = []

    # H：從右往左，兩兩配對「跌破5MA」事件
    # 例如事件 [2, 8, 15, 23]：
    # 先配 23-15，再配 15-8，再配 8-2
    for k in range(len(down_breaks) - 1, 0, -1):
        right_idx = down_breaks[k]
        left_idx = down_breaks[k - 1]

        segment = range(left_idx, right_idx + 1)
        h_idx = min(segment, key=lambda j: candles[j]["y_high"])  # y越小，價格越高

        labels.append(
            {
                "idx": h_idx,
                "type": "H",
                "left_idx": left_idx,
                "right_idx": right_idx,
                "source": "跌破5MA",
            }
        )

    # L：從右往左，兩兩配對「突破5MA」事件
    for k in range(len(up_breaks) - 1, 0, -1):
        right_idx = up_breaks[k]
        left_idx = up_breaks[k - 1]

        segment = range(left_idx, right_idx + 1)
        l_idx = max(segment, key=lambda j: candles[j]["y_low"])  # y越大，價格越低

        labels.append(
            {
                "idx": l_idx,
                "type": "L",
                "left_idx": left_idx,
                "right_idx": right_idx,
                "source": "突破5MA",
            }
        )

    # 畫圖用：由左到右畫，避免圖層順序混亂
    labels_for_draw = sorted(labels, key=lambda x: (x["idx"], x["type"]))

    # 表格用：由右到左看，符合使用者邏輯
    labels_for_table = sorted(labels, key=lambda x: x["right_idx"], reverse=True)

    return labels_for_draw, labels_for_table, down_breaks, up_breaks


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
# 畫標記
# =========================
def draw_labels(
    img: Image.Image,
    candles,
    labels_for_draw,
    crop_y0: int,
    crop_y1: int,
    display_mode: str = "HL",
    draw_box: bool = True,
):
    out = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)

    use_chinese = display_mode == "頭底"

    for item in labels_for_draw:
        idx = item["idx"]
        typ = item["type"]
        c = candles[idx]

        if display_mode == "HL":
            text = "H" if typ == "H" else "L"
        else:
            text = "頭" if typ == "H" else "底"

        # 嚴格遵守：圓形直徑不可超過K棒寬度
        diam = max(3, int(c["width"]))
        radius = diam / 2

        font_size = max(6, diam)
        font = get_font(font_size, prefer_chinese=use_chinese)

        x = c["x"]

        if typ == "H":
            y = max(crop_y0 + 2, c["y_high"] - diam - 3)
            text_color = (255, 0, 0, 255)
            box_color = (255, 0, 0, 255)
        else:
            y = min(crop_y1 - diam - 3, c["y_low"] + 3)
            text_color = (0, 180, 0, 255)
            box_color = (0, 180, 0, 255)

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
            width=max(1, int(diam * 0.18)),
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
# 主處理函式
# =========================
def annotate_kline_image(
    img: Image.Image,
    crop_top_ratio=0.30,
    crop_bottom_ratio=0.65,
    crop_left_ratio=0.00,
    crop_right_ratio=0.89,
    tolerance_px=6,
    display_mode="HL",
    draw_box=True,
    price_top=None,
    price_bottom=None,
    start_date=None,
    end_date=None,
):
    img = img.convert("RGB")
    W, H = img.size

    x0 = int(W * crop_left_ratio)
    x1 = int(W * crop_right_ratio)
    y0 = int(H * crop_top_ratio)
    y1 = int(H * crop_bottom_ratio)

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

        # y 越大代表價格越低
        # rel > 0：收盤在 5MA 下方
        # rel < 0：收盤在 5MA 上方
        c["rel"] = float(c["close_y"] - c["ma_y"])

    labels_for_draw, labels_for_table, down_breaks, up_breaks = build_labels_right_to_left(
        candles,
        tolerance_px=tolerance_px,
    )

    result = draw_labels(
        img=img,
        candles=candles,
        labels_for_draw=labels_for_draw,
        crop_y0=y0,
        crop_y1=y1,
        display_mode=display_mode,
        draw_box=draw_box,
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
                "事件類型": item["source"],
                "標記K棒序號": idx + 1,
                "右側基準點序號": item["right_idx"] + 1,
                "左側基準點序號": item["left_idx"] + 1,
                "區間": f'{item["left_idx"] + 1} ~ {item["right_idx"] + 1}',
                "估算日期": est_date,
                "估算價格": "" if price is None else round(price, 2),
                "K棒顏色": "紅K" if c["color"] == "red" else "綠K",
            }
        )

    info = {
        "candles": len(candles),
        "down_breaks": len(down_breaks),
        "up_breaks": len(up_breaks),
        "labels": len(labels_for_draw),
        "crop_box": (x0, y0, x1, y1),
        "crop_img": crop_img,
        "down_break_indices": [i + 1 for i in down_breaks],
        "up_break_indices": [i + 1 for i in up_breaks],
    }

    return result, rows, info


# =========================
# Streamlit UI
# =========================
st.title("K線頭部 / 底部 自動標記")
st.caption("修正版：依『從最右邊往左邊回推』規則，配對突破 / 跌破 5MA 事件。")

with st.sidebar:
    st.header("標示設定")

    display_mode = st.radio(
        "標示模式",
        ["HL", "頭底"],
        index=0,
        help="HL = H/L；頭底 = 頭/底",
    )

    draw_box = st.checkbox("圈出被判定的K棒", value=True)

    st.divider()
    st.header("主圖裁切")

    crop_top_ratio = st.slider("主圖上緣", 0.00, 0.85, DEFAULTS["crop_top_ratio"], 0.01)
    crop_bottom_ratio = st.slider("主圖下緣", 0.20, 1.00, DEFAULTS["crop_bottom_ratio"], 0.01)
    crop_left_ratio = st.slider("主圖左邊界", 0.00, 0.40, DEFAULTS["crop_left_ratio"], 0.01)
    crop_right_ratio = st.slider("主圖右邊界", 0.50, 1.00, DEFAULTS["crop_right_ratio"], 0.01)

    st.divider()
    st.header("判定設定")

    tolerance_px = st.slider(
        "貼線容許誤差",
        0,
        20,
        DEFAULTS["tolerance_px"],
        1,
        help="避免收盤價剛好貼近5MA時被過度判定。通常 4~8 比較合理。",
    )

    st.divider()
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

    x0 = int(W * crop_left_ratio)
    x1 = int(W * crop_right_ratio)
    y0 = int(H * crop_top_ratio)
    y1 = int(H * crop_bottom_ratio)

    preview = draw_main_crop_box(img, x0, y0, x1, y1)

    tab1, tab2, tab3, tab4 = st.tabs(["結果", "主圖裁切檢查", "事件明細", "原圖"])

    try:
        result, rows, info = annotate_kline_image(
            img=img,
            crop_top_ratio=crop_top_ratio,
            crop_bottom_ratio=crop_bottom_ratio,
            crop_left_ratio=crop_left_ratio,
            crop_right_ratio=crop_right_ratio,
            tolerance_px=tolerance_px,
            display_mode=display_mode,
            draw_box=draw_box,
            price_top=price_top,
            price_bottom=price_bottom,
            start_date=start_date,
            end_date=end_date,
        )

        with tab1:
            st.subheader("標記結果")
            st.image(result, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("K棒數", info["candles"])
            c2.metric("跌破5MA事件數", info["down_breaks"])
            c3.metric("突破5MA事件數", info["up_breaks"])
            c4.metric("標記總數", info["labels"])

            st.download_button(
                "下載標記圖 PNG",
                data=pil_to_png_bytes(result),
                file_name=f"marked_{display_mode}_right_to_left.png",
                mime="image/png",
            )

            if rows:
                st.subheader("標記明細（由右往左排序）")
                st.dataframe(rows, use_container_width=True)
            else:
                st.warning("目前沒有形成兩個同類基準點，因此沒有標記。")

        with tab2:
            st.subheader("主圖裁切檢查")
            st.caption("橘框應該只包住主K線圖，不要包到成交量、KD、MACD。")
            st.image(preview, use_container_width=True)

            st.subheader("實際裁切區")
            st.image(info["crop_img"], use_container_width=True)

        with tab3:
            st.subheader("突破 / 跌破事件序號")
            st.write("跌破5MA事件序號：", info["down_break_indices"])
            st.write("突破5MA事件序號：", info["up_break_indices"])

            if rows:
                st.subheader("標記區間明細")
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

else:
    st.info("請先上傳一張 K 線截圖。")
