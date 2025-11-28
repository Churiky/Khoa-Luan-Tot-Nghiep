# app/signals.py
import os
import numpy as np
import pandas as pd
from .analysis import get_predict_path, analyze_monthly, analyze_quarterly


# -----------------------
# Helpers
# -----------------------
def normalize_pred(df):
    """Chuẩn hóa file dự đoán: đổi 'Ngày' → 'date'"""
    if "Ngày" in df.columns:
        df = df.rename(columns={"Ngày": "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    return df.sort_values("date")


def ma(series, window):
    return series.rolling(window).mean()


def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    ma_up = up.ewm(alpha=1/period, min_periods=period).mean()
    ma_down = down.ewm(alpha=1/period, min_periods=period).mean()

    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def signal_recommend(pred_path):
    """
    Hàm tính toán tín hiệu mua/bán dựa trên dự đoán giá đóng cửa.
    Luôn trả về một signal hợp lệ: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    """
    # Kiểm tra file
    if not os.path.exists(pred_path):
        return {"ok": True, "signal": "HOLD", "score": 0.0, "note": "Chưa có file dự đoán"}

    try:
        df = pd.read_csv(pred_path)
        df = normalize_pred(df)

        if "Giá_đóng_cửa_dự_đoán" not in df.columns:
            return {"ok": True, "signal": "HOLD", "score": 0.0, "note": "Thiếu cột Giá_đóng_cửa_dự_đoán"}

        close = df["Giá_đóng_cửa_dự_đoán"].astype(float)

        if len(close) < 21:
            return {"ok": True, "signal": "HOLD", "score": 0.0, "note": "Cần >= 21 ngày để tính MA20 & RSI"}

        # Tính MA và RSI
        ma5 = ma(close, 5)
        ma20 = ma(close, 20)
        rsi14 = rsi(close, 14)

        latest_ma5 = float(ma5.iloc[-1])
        latest_ma20 = float(ma20.iloc[-1])
        latest_rsi = float(rsi14.iloc[-1])

        vol = float(close.pct_change().rolling(10).std().iloc[-1] or 0.0)

        # Tính score
        score = 0.0
        score += 0.4 if latest_ma5 > latest_ma20 else -0.2
        score += 0.3 if latest_rsi < 30 else (-0.3 if latest_rsi > 70 else 0)
        score -= min(0.2, vol * 2)
        score = max(-1.0, min(1.0, score))

        # Chọn tín hiệu
        if score > 0.35:
            signal = "STRONG_BUY"
        elif score > 0.05:
            signal = "BUY"
        elif score < -0.35:
            signal = "STRONG_SELL"
        elif score < -0.05:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "ok": True,
            "signal": signal,
            "score": round(score, 3),
            "note": "Tính toán MA & RSI thành công"
        }

    except Exception as e:
        return {"ok": True, "signal": "HOLD", "score": 0.0, "note": f"Lỗi khi tính: {str(e)}"}
# -----------------------
# Simple Signal
# -----------------------
def signal_simple(pred_path, horizon=7):
    if not os.path.exists(pred_path):
        return {"ok": False, "msg": "Chưa có file dự đoán"}

    df = pd.read_csv(pred_path)
    df = normalize_pred(df)

    if "Giá_đóng_cửa_dự_đoán" not in df.columns:
        return {"ok": False, "msg": "Thiếu cột Giá_đóng_cửa_dự_đoán"}

    close = df["Giá_đóng_cửa_dự_đoán"].astype(float).reset_index(drop=True)

    if len(close) < horizon + 1:
        return {"ok": False, "msg": f"Cần ít nhất {horizon+1} ngày dự đoán"}

    last = close.iloc[-(horizon + 1):].reset_index(drop=True)
    start = float(last.iloc[0])
    end = float(last.iloc[-1])

    pct = (end - start) / (start + 1e-9)
    vol = float(last.pct_change().abs().mean())

    conf = min(1.0, max(0.0, abs(pct) / (0.02 + vol)))

    if pct > 0.01:
        signal = "BUY"
        trend = "up"
    elif pct < -0.01:
        signal = "SELL"
        trend = "down"
    else:
        signal = "HOLD"
        trend = "flat"

    return {
        "ok": True,
        "trend": trend,
        "signal": signal,
        "confidence": round(conf, 3),
        "pct_change": round(pct, 4),
        "note": f"Xét {horizon} ngày gần nhất"
    }


# -----------------------
# Advanced Signal
# -----------------------
def signal_advanced(pred_path):
    """
    Tính toán tín hiệu dựa trên MA5, MA20 và RSI14.
    Luôn trả về 'signal' và 'score' hợp lệ, kể cả khi dữ liệu thiếu hoặc lỗi.
    """
    try:
        # File không tồn tại
        if not os.path.exists(pred_path):
            return {"ok": True, "signal": "HOLD", "score": 0.0, "note": "Chưa có file dự đoán"}

        # Đọc CSV
        df = pd.read_csv(pred_path)

        # Chuẩn hóa dữ liệu (nếu có)
        if "normalize_pred" in globals():
            df = normalize_pred(df)

        if "Giá_đóng_cửa_dự_đoán" not in df.columns:
            return {"ok": True, "signal": "HOLD", "score": 0.0, "note": "Thiếu cột Giá_đóng_cửa_dự_đoán"}

        close = df["Giá_đóng_cửa_dự_đoán"].astype(float).dropna()

        # Cần đủ 21 ngày để tính MA20 & RSI14
        if len(close) < 21:
            return {"ok": True, "signal": "HOLD", "score": 0.0, "note": "Cần >= 21 ngày để tính MA20 & RSI"}

        # Tính MA và RSI
        ma5 = ma(close, 5).fillna(0.0)
        ma20 = ma(close, 20).fillna(0.0)
        rsi14 = rsi(close, 14).fillna(50.0)  # Nếu NaN, coi RSI = 50 trung lập

        latest_ma5 = float(ma5.iloc[-1] or 0.0)
        latest_ma20 = float(ma20.iloc[-1] or 0.0)
        latest_rsi = float(rsi14.iloc[-1] or 50.0)

        prev_ma5 = float(ma5.iloc[-2] or 0.0)
        prev_ma20 = float(ma20.iloc[-2] or 0.0)

        # Kiểm tra MA cross
        if (prev_ma5 <= prev_ma20) and (latest_ma5 > latest_ma20):
            ma_cross = "MA5_cross_up"
        elif (prev_ma5 >= prev_ma20) and (latest_ma5 < latest_ma20):
            ma_cross = "MA5_cross_down"
        else:
            ma_cross = "no_cross"

        # Độ biến động
        vol = float(close.pct_change().rolling(10).std().iloc[-1] or 0.0)

        # Tính score
        score = 0.0
        score += 0.4 if latest_ma5 > latest_ma20 else -0.2
        if latest_rsi < 30:
            score += 0.3
        elif latest_rsi > 70:
            score -= 0.3
        score -= min(0.2, vol * 2)

        # Bảo vệ NaN
        if np.isnan(score):
            score = 0.0

        # Giới hạn score
        score = max(-1.0, min(1.0, score))

        # Gán tín hiệu dựa trên score
        if score > 0.35:
            signal = "STRONG_BUY"
        elif score > 0.05:
            signal = "BUY"
        elif score < -0.35:
            signal = "STRONG_SELL"
        elif score < -0.05:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "ok": True,
            "rsi": round(latest_rsi, 2),
            "ma5": round(latest_ma5, 4),
            "ma20": round(latest_ma20, 4),
            "ma_cross": ma_cross,
            "volatility": round(vol, 6),
            "score": round(score, 3),
            "signal": signal,
            "note": "Tính toán MA & RSI"
        }

    except Exception as e:
        # Nếu có lỗi bất ngờ, vẫn trả HOLD và score = 0
        return {"ok": True, "signal": "HOLD", "score": 0.0, "note": f"Lỗi khi tính: {str(e)}"}



# -----------------------
# Summary (Tổng hợp)
# -----------------------
def signal_summary(base_dir, file):
    pred_path = get_predict_path(base_dir, file)

    if not os.path.exists(pred_path):
        return {"ok": False, "msg": "Chưa có file dự đoán"}

    monthly, err_m = analyze_monthly(pred_path)
    quarterly, err_q = analyze_quarterly(pred_path)

    adv = signal_advanced(pred_path)
    simp = signal_simple(pred_path, horizon=30)

    if not adv.get("ok", True):
        return {"ok": False, "msg": adv.get("msg", "Lỗi advanced")}

    score = adv.get("score", 0.0)

    # Thêm xu hướng tháng
    try:
        if monthly and len(monthly) >= 2:
            last = float(monthly[-1]["Trung_bình"])
            prev = float(monthly[-2]["Trung_bình"])
            monthly_pct = (last - prev) / (prev + 1e-9)
            score += max(-0.3, min(0.3, monthly_pct))
    except:
        pass

    score = max(-1.0, min(1.0, score))

    if score > 0.4:
        overall = "STRONG_BUY"
    elif score > 0.1:
        overall = "BUY"
    elif score < -0.4:
        overall = "STRONG_SELL"
    elif score < -0.1:
        overall = "SELL"
    else:
        overall = "HOLD"

    return {
        "ok": True,
        "overall_signal": overall,
        "score": round(score, 3),
        "advanced": adv,
        "simple_30d": simp,
        "monthly": monthly,
        "quarterly": quarterly,
        "note": "Kết hợp MA/RSI và xu hướng tháng"
    }
