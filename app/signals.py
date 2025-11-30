import os
import pandas as pd
import numpy as np


# ============================================================
#                   🔧 UTILITY & INDICATORS
# ============================================================

def load_pred_file(pred_path):
    """Load file + chuẩn hóa cột ngày"""
    if not os.path.exists(pred_path):
        return None, "File dự đoán không tồn tại"

    df = pd.read_csv(pred_path)

    # Chuẩn hóa cột ngày
    if "Ngày" in df.columns:
        df["date"] = pd.to_datetime(df["Ngày"], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        return None, "Không tìm thấy cột Ngày"

    df = df.dropna(subset=["date"]).sort_values("date")
    return df, None


def ma(series, window):
    return series.rolling(window).mean()


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta.clip(upper=0)).abs()

    avg_up = up.rolling(period).mean()
    avg_down = down.rolling(period).mean()

    rs = avg_up / (avg_down + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(close):
    """MACD tiêu chuẩn 12-26-9"""
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd_line = ema12 - ema26
    signal_line = ema(macd_line, 9)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_roc(close, period=10):
    return close.pct_change(period) * 100


# ============================================================
#                🚀 SIMPLE SIGNAL (NHANH)
# ============================================================

def signal_simple(pred_path, horizon=7):
    df, err = load_pred_file(pred_path)
    if err: 
        return {"ok": False, "msg": err}

    if "Giá_đóng_cửa_dự_đoán" not in df.columns:
        return {"ok": False, "msg": "Thiếu cột Giá_đóng_cửa_dự_đoán"}

    close = df["Giá_đóng_cửa_dự_đoán"].astype(float)

    if len(close) < 2:
        return {"ok": True, "signal": "HOLD", "confidence": 0}

    horizon = min(horizon, len(close) - 1)
    pct_change = (close.iloc[-1] - close.iloc[-horizon]) / (close.iloc[-horizon] + 1e-9)

    # Logic đơn giản
    if pct_change > 0.01:
        signal = "BUY"
    elif pct_change < -0.01:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "ok": True,
        "signal": signal,
        "pct_change": float(pct_change)
    }


# ============================================================
#          🚀 ADVANCED SIGNAL – PRO LEVEL
# ============================================================

def signal_advanced(pred_path):
    df, err = load_pred_file(pred_path)
    if err:
        return {"ok": False, "msg": err}

    if "Giá_đóng_cửa_dự_đoán" not in df.columns:
        return {"ok": False, "msg": "Thiếu cột Giá_đóng_cửa_dự_đoán"}

    close = df["Giá_đóng_cửa_dự_đoán"].astype(float)

    # ===============================
    #     🔥 TÍNH TOÀN BỘ CHỈ BÁO
    # ===============================
    df["MA7"] = ma(close, 7)
    df["MA20"] = ma(close, 20)
    df["MA50"] = ma(close, 50)

    df["RSI14"] = compute_rsi(close, 14)

    macd_line, signal_line, histogram = compute_macd(close)
    df["MACD"] = histogram

    df["ROC10"] = compute_roc(close, 10)

    # Output theo ngày cho giao diện
    daily = df.apply(lambda r: {
        "date": r["date"].strftime("%Y-%m-%d"),
        "close": float(r["Giá_đóng_cửa_dự_đoán"]),
        "ma7": None if pd.isna(r["MA7"]) else float(r["MA7"]),
        "ma20": None if pd.isna(r["MA20"]) else float(r["MA20"]),
        "ma50": None if pd.isna(r["MA50"]) else float(r["MA50"]),
        "rsi": None if pd.isna(r["RSI14"]) else float(r["RSI14"]),
        "macd": None if pd.isna(r["MACD"]) else float(r["MACD"]),
        "roc": None if pd.isna(r["ROC10"]) else float(r["ROC10"]),
    }, axis=1).tolist()

    # ===============================
    #       🔥 TÍNH SIGNAL CUỐI
    # ===============================
    last = df.iloc[-1]

    score = 0
    reasons = []

    # MA trend
    if last["MA7"] > last["MA20"] > last["MA50"]:
        score += 0.4; reasons.append("MA Uptrend mạnh")
    elif last["MA7"] > last["MA20"]:
        score += 0.2; reasons.append("MA Uptrend")
    elif last["MA7"] < last["MA20"] < last["MA50"]:
        score -= 0.4; reasons.append("MA Downtrend mạnh")
    else:
        score -= 0.1

    # RSI
    if last["RSI14"] < 30:
        score += 0.3; reasons.append("RSI Quá bán")
    elif last["RSI14"] > 70:
        score -= 0.3; reasons.append("RSI Quá mua")

    # MACD
    if last["MACD"] > 0:
        score += 0.2; reasons.append("MACD dương")
    else:
        score -= 0.2; reasons.append("MACD âm")

    # ROC (momentum)
    if last["ROC10"] > 3:
        score += 0.2; reasons.append("Momentum mạnh")
    elif last["ROC10"] < -3:
        score -= 0.2; reasons.append("Momentum giảm mạnh")

    # Normalize score
    score = max(-1, min(1, score))

    # Decision
    if score >= 0.5: final = "STRONG_BUY"
    elif score >= 0.2: final = "BUY"
    elif score <= -0.5: final = "STRONG_SELL"
    elif score <= -0.2: final = "SELL"
    else: final = "HOLD"

    return {
        "ok": True,
        "overall_signal": final,
        "score": float(round(score, 3)),
        "reasons": reasons,
        "daily": daily
    }


# ============================================================
#            🚀 SUMMARY + RECOMMEND (PLACEHOLDER)
# ============================================================

def signal_recommend(pred_path):
    return {"ok": True, "signal": "HOLD", "score": 0.0}


def signal_summary(base_dir, file):
    """
    Summary = kết hợp Advanced + Recommend
    FIX:
    - Đọc đúng file path tạm
    - Trả về daily cho dashboard
    - Tính điểm trung bình rõ ràng
    """
    try:
        pred_path = os.path.join(base_dir, "DATA_PREDICT", file)

        if not os.path.exists(pred_path):
            return {"ok": False, "msg": "File dự đoán không tồn tại"}

        # --- gọi bản advanced ---
        adv = signal_advanced(pred_path)
        if not adv.get("ok", True):
            return adv

        # --- gọi bản recommend ---
        rec = signal_recommend(pred_path)

        # --- điểm từng phần ---
        adv_score = float(adv.get("score", 0.0))
        rec_score = float(rec.get("score", 0.0))

        # --- kết hợp với trọng số ---
        combined_score = (adv_score * 0.75) + (rec_score * 0.25)
        combined_score = max(-1, min(1, combined_score))

        # --- quyết định tín hiệu ---
        if combined_score >= 0.5:
            overall = "STRONG_BUY"
        elif combined_score >= 0.2:
            overall = "BUY"
        elif combined_score <= -0.5:
            overall = "STRONG_SELL"
        elif combined_score <= -0.2:
            overall = "SELL"
        else:
            overall = "HOLD"

        result = {
            "ok": True,
            "overall_signal": overall,
            "score": round(combined_score, 3),
            "components": {
                "advanced": adv,
                "recommend": rec
            }
        }

        # 🔥 Quan trọng: Forward daily ra UI
        if isinstance(adv.get("daily"), list):
            result["daily"] = adv["daily"]

        return result

    except Exception as e:
        return {"ok": False, "msg": str(e)}

