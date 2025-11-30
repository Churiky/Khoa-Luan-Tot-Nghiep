import time
import requests
import pandas as pd
import yfinance as yf
from .config import api_keys


# ============================
# FINNHUB
# ============================
def _try_finnhub(symbol):
    key = api_keys.FINNHUB_KEY
    if not key:
        return None
    try:
        import datetime
        to_ts = int(time.time())
        frm_ts = int((datetime.datetime.now() - datetime.timedelta(days=30000)).timestamp())

        url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&from={frm_ts}&to={to_ts}&token={key}"
        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            return None

        data = r.json()
        if data.get("s") != "ok":
            return None

        df = pd.DataFrame({
            "date": pd.to_datetime(data["t"], unit="s"),
            "open": data["o"],
            "high": data["h"],
            "low": data["l"],
            "close": data["c"],
            "volume": data["v"]
        })

        df = df.sort_values("date").reset_index(drop=True)
        return df
    except:
        return None


# ============================
# ALPHAVANTAGE
# ============================
def _try_alphavantage(symbol):
    key = api_keys.AV_KEY
    if not key:
        return None
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": key
        }

        r = requests.get(url, params=params, timeout=10)
        j = r.json()

        if "Time Series (Daily)" not in j:
            return None

        rows = []
        ts = j["Time Series (Daily)"]

        for d, v in ts.items():
            rows.append({
                "date": pd.to_datetime(d),
                "open": float(v["1. open"]),
                "high": float(v["2. high"]),
                "low": float(v["3. low"]),
                "close": float(v["4. close"]),
                "volume": float(v.get("6. volume", 0))
            })

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        return df

    except:
        return None

# ============================
# YFINANCE
# ============================
def _try_yfinance(symbol):
    try:
        t = yf.Ticker(symbol)
        df = t.history(period="max", interval="1d")

        if df is None or df.empty:
            return None

        df = df.reset_index()

        df = df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })

        df = df.dropna()
        return df[["date", "open", "high", "low", "close", "volume"]]

    except:
        return None


# ============================
# MAIN — TRY MULTIPLE SOURCES
# ============================
def download_stock(symbol: str):
    symbol = symbol.upper()

    for fn, name in [
        (_try_finnhub, "finnhub"),
        (_try_alphavantage, "alphavantage"),
        (_try_yfinance, "yfinance"),
    ]:
        df = fn(symbol)
        if df is not None:

            # ==============================
            #  🔥 FIX TZ DATETIME ERROR 🔥
            # ==============================
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

            # rename format giống VNM_data.csv bạn đang dùng
            df = df.rename(columns={"date": "time"})

            # required column
            df["symbol"] = symbol

            # reorder đúng chuẩn training
            df = df[["time", "open", "high", "low", "close", "volume", "symbol"]]

            return df, name

    return None, None
