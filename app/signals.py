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
def _compute_indicators_and_score(df):
    """Input df must contain 'Giá_đóng_cửa_dự_đoán' and 'date' (datetime)"""
    # ensure sorted
    df = df.sort_values("date").reset_index(drop=True)

    # cast
    df["Giá_đóng_cửa_dự_đoán"] = pd.to_numeric(df["Giá_đóng_cửa_dự_đoán"], errors="coerce")
    close = df["Giá_đóng_cửa_dự_đoán"].fillna(method="ffill").fillna(method="bfill")

    # indicators
    df["MA7"]  = ma(close, 7)
    df["MA14"] = ma(close, 14)
    df["MA50"] = ma(close, 50)
    df["RSI14"] = compute_rsi(close, 14)
    _, _, hist = compute_macd(close)
    df["MACD"] = hist
    df["ROC10"] = compute_roc(close, 10)

    # prepare daily
    daily = []
    for _, r in df.iterrows():
        daily.append({
            "date": r["date"].strftime("%Y-%m-%d"),
            "close": None if pd.isna(r["Giá_đóng_cửa_dự_đoán"]) else float(r["Giá_đóng_cửa_dự_đoán"]),
            "ma7": None if pd.isna(r["MA7"]) else float(r["MA7"]),
            "ma14": None if pd.isna(r["MA14"]) else float(r["MA14"]),
            "ma50": None if pd.isna(r["MA50"]) else float(r["MA50"]),
            "rsi": None if pd.isna(r["RSI14"]) else float(r["RSI14"]),
            "macd": None if pd.isna(r["MACD"]) else float(r["MACD"]),
            "roc": None if pd.isna(r["ROC10"]) else float(r["ROC10"]),
        })

    # compute score & reasons based on last available row that has close value
    # find last valid row index
    valid_idx = df["Giá_đóng_cửa_dự_đoán"].last_valid_index()
    if valid_idx is None:
        # no data
        return {"daily": daily, "overall_signal": "HOLD", "score": 0.0, "reasons": []}

    last = df.loc[valid_idx]

    score = 0.0
    reasons = []

    # MA Trend (use MA7, MA14, MA50)
    try:
        if not pd.isna(last["MA7"]) and not pd.isna(last["MA14"]) and not pd.isna(last["MA50"]):
            if last["MA7"] > last["MA14"] > last["MA50"]:
                score += 0.4; reasons.append("MA Uptrend mạnh")
            elif last["MA7"] > last["MA14"]:
                score += 0.2; reasons.append("MA Uptrend")
            elif last["MA7"] < last["MA14"] < last["MA50"]:
                score -= 0.4; reasons.append("MA Downtrend mạnh")
            else:
                score -= 0.1
    except Exception:
        pass

    # RSI
    try:
        if not pd.isna(last["RSI14"]):
            if last["RSI14"] < 30:
                score += 0.3; reasons.append("RSI Quá bán")
            elif last["RSI14"] > 70:
                score -= 0.3; reasons.append("RSI Quá mua")
    except Exception:
        pass

    # MACD
    try:
        if not pd.isna(last["MACD"]):
            if last["MACD"] > 0:
                score += 0.2; reasons.append("MACD dương")
            else:
                score -= 0.2; reasons.append("MACD âm")
    except Exception:
        pass

    # ROC
    try:
        if not pd.isna(last["ROC10"]):
            if last["ROC10"] > 3:
                score += 0.2; reasons.append("Momentum mạnh")
            elif last["ROC10"] < -3:
                score -= 0.2; reasons.append("Momentum giảm mạnh")
    except Exception:
        pass

    score = max(-1.0, min(1.0, score))

    if score >= 0.5:
        overall = "STRONG_BUY"
    elif score >= 0.2:
        overall = "BUY"
    elif score <= -0.5:
        overall = "STRONG_SELL"
    elif score <= -0.2:
        overall = "SELL"
    else:
        overall = "HOLD"

    return {"daily": daily, "overall_signal": overall, "score": float(round(score, 4)), "reasons": reasons}
def signal_recommend(pred_path):
    """
    Khuyến nghị tổng hợp dựa trên:
    - Technical score (MA/RSI/MACD/ROC)
    - Momentum (ROC3/ROC5)
    - Trend persistence (slope MA50)
    - Prediction consistency (corr hoặc độ mượt)
    - Risk (volatility)
    """
    try:
        df, err = load_pred_file(pred_path)
        if err:
            return {"ok": False, "msg": err}

        col = "Giá_đóng_cửa_dự_đoán"
        if col not in df.columns:
            return {"ok": False, "msg": f"Thiếu cột {col}"}

        # Numeric + fill
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if len(df) < 10:
            return {"ok": False, "msg": "Không đủ dữ liệu để khuyến nghị"}

        # -----------------------------
        # 1. ADVANCED (Technical)
        # -----------------------------
        adv = _compute_indicators_and_score(df)
        adv_score = float(adv.get("score", 0.0))

        closes = df[col].fillna(method="ffill").fillna(method="bfill")
        n = len(closes)

        # -----------------------------
        # 2. MOMENTUM (ROC3 + ROC5)
        # -----------------------------
        def roc(window):
            if n <= window:
                return 0.0
            return float(closes.pct_change(window).iloc[-1] * 100)

        roc3 = roc(3)
        roc5 = roc(5)

        # Normalize: ±5% → [-1..1]
        mom_score = max(-1.0, min(1.0, (roc3 + roc5) / 2 / 5))

        # -----------------------------
        # 3. TREND persistence (slope MA50)
        # -----------------------------
        df["MA50"] = ma(closes, 50)
        slope = 0.0
        try:
            last = df["MA50"].last_valid_index()
            if last is not None and last >= 7:
                v1 = df.loc[last, "MA50"]
                v0 = df.loc[last - 7, "MA50"]
                if not pd.isna(v1) and not pd.isna(v0):
                    slope = (v1 - v0) / (abs(v0) + 1e-9)
        except:
            slope = 0.0

        # MAP slope ~ 0.01 → score ~ 0.5
        trend_score = max(-1.0, min(1.0, slope * 50))

        # -----------------------------
        # 4. Prediction Consistency
        # -----------------------------
        actual_col = None
        for c in ["Giá_đóng_cửa_thực_tế", "actual", "real"]:
            if c in df.columns:
                actual_col = c
                break

        if actual_col:
            seg = df[[actual_col, col]].dropna().iloc[-20:]
            if len(seg) >= 5:
                corr = seg[actual_col].corr(seg[col])
                pred_conf = float(corr if not pd.isna(corr) else 0.0)
            else:
                pred_conf = 0.0
        else:
            # Dữ liệu không có giá thực → đánh giá độ mượt
            recent = closes.iloc[-20:] if n >= 20 else closes
            rel_std = recent.std() / (recent.mean() + 1e-9)
            pred_conf = max(-1.0, min(1.0, (0.05 - rel_std) / 0.05))

        # -----------------------------
        # 5. RISK (Volatility)
        # -----------------------------
        returns = closes.pct_change().dropna()
        if len(returns) > 0:
            vol = returns.rolling(14).std().iloc[-1] if len(returns) >= 14 else returns.std()
        else:
            vol = 0.0

        # map vol 0.005 → 0, 0.05 → 1
        risk_norm = max(0.0, min(1.0, (vol - 0.005) / (0.05 - 0.005)))
        risk_score = -risk_norm  # risk giảm điểm

        # -----------------------------
        # 6. COMPOSITE SCORE
        # -----------------------------
        w_adv = 0.50
        w_mom = 0.12
        w_trend = 0.13
        w_conf = 0.15
        w_risk = 0.10

        combined = (
            adv_score * w_adv +
            mom_score * w_mom +
            trend_score * w_trend +
            pred_conf * w_conf +
            risk_score * w_risk
        )

        combined = max(-1.0, min(1.0, combined))

        # -----------------------------
        # 7. MAP → LABEL
        # -----------------------------
        if combined >= 0.65:
            overall = "STRONG_BUY"
        elif combined >= 0.30:
            overall = "BUY"
        elif combined <= -0.65:
            overall = "STRONG_SELL"
        elif combined <= -0.30:
            overall = "SELL"
        else:
            overall = "HOLD"

        # -----------------------------
        # 8. REASONS
        # -----------------------------
        reasons = []
        if adv_score >= 0.3: reasons.append("Technical MA/RSI/MACD ủng hộ mua")
        if adv_score <= -0.3: reasons.append("Technical MA/RSI/MACD ủng hộ bán")

        if mom_score > 0.1: reasons.append(f"Momentum tích cực ({roc3:.2f}%, {roc5:.2f}%)")
        if mom_score < -0.1: reasons.append(f"Momentum tiêu cực ({roc3:.2f}%, {roc5:.2f}%)")

        if trend_score > 0.1: reasons.append("Xu hướng MA50 dốc lên")
        if trend_score < -0.1: reasons.append("Xu hướng MA50 dốc xuống")

        if pred_conf > 0.3: reasons.append("Dự đoán khớp/ổn định")
        if pred_conf < -0.3: reasons.append("Dự đoán không ổn định")

        if risk_score < -0.25: reasons.append("Biến động cao (rủi ro tăng)")

        return {
            "ok": True,
            "overall_signal": overall,
            "score": float(round(combined, 4)),
            "reasons": reasons,
            "components": {
                "advanced": adv_score,
                "momentum": mom_score,
                "trend": trend_score,
                "pred_conf": pred_conf,
                "risk": risk_score
            },
            "daily": adv.get("daily", [])
        }

    except Exception as e:
        return {"ok": False, "msg": str(e)}


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

