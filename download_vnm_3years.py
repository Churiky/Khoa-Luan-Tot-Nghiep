# download_vnm_3years_debug.py
import os
import sys
import traceback
import datetime
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import timedelta

# Thư mục lưu data/debug
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "DATA")
os.makedirs(DATA_DIR, exist_ok=True)

def save_df(df, name):
    path = os.path.join(DATA_DIR, name)
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Saved: {path}")
    except Exception as e:
        print(f"Error saving {name}: {e}")

# --- Hàm lấy dữ liệu dài hạn Yahoo (chia nhỏ để tránh giới hạn 500 ngày) ---
def fetch_long_history(symbol, years=10, interval="1d", chunk_days=500):
    end_date = datetime.date.today()
    start_date = end_date - timedelta(days=365 * years)

    df_list = []
    chunk_start = start_date

    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end_date)
        try:
            print(f"Fetching {chunk_start} -> {chunk_end}")
            data = yf.download(symbol, start=chunk_start, end=chunk_end, interval=interval)
            if not data.empty:
                df_list.append(data)
        except Exception as e:
            print(f"Yahoo fetch error {chunk_start} -> {chunk_end}: {e}")
        chunk_start = chunk_end

    if not df_list:
        return None

    df = pd.concat(df_list)
    df = df[~df.index.duplicated(keep='first')]
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col if c]) for col in df.columns.values]

    return df

def get_yahoo():
    print("\n--- Yahoo Finance ---")
    try:
        df = fetch_long_history("VNM.VN", years=10, interval="1d", chunk_days=500)
        if df is None:
            print("Yahoo: no data returned")
            return None
        print("Yahoo: rows=", len(df))
        print(df.head().to_string(index=False))
        save_df(df, "VNM_Yahoo_debug.csv")
        return df
    except Exception as e:
        print("Yahoo error:", e)
        traceback.print_exc()
        return None

# --- Thử các cách gọi vnstock ---
def try_vnstock_methods():
    print("\n--- Vnstock attempts ---")
    attempts = []
    try:
        import vnstock
        print("vnstock package version (vnstock.__version__ if available):", getattr(vnstock, "__version__", "unknown"))
    except Exception as e:
        print("Import vnstock failed:", e)
        traceback.print_exc()
        return attempts

    from vnstock import Vnstock
    vn = Vnstock()

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=10*365)  # 10 năm
    s = start_date.strftime("%Y-%m-%d")
    e = end_date.strftime("%Y-%m-%d")
    print("Requested range:", s, "->", e)

    def attempt(name, fn):
        print(f"\nAttempt: {name}")
        try:
            df = fn()
            if df is None:
                print("Result: None")
                attempts.append((name, None, "None"))
                return None
            df = pd.DataFrame(df)
            print("rows:", len(df))
            print("columns:", df.columns.tolist())
            print("dtypes:\n", df.dtypes)
            print("head:\n", df.head().to_string(index=False))
            filename = f"VNM_vnstock_{name.replace(' ','_')}.csv"
            save_df(df, filename)
            attempts.append((name, df, None))
            return df
        except Exception as ex:
            print(f"Exception in attempt {name}: {ex}")
            traceback.print_exc()
            attempts.append((name, None, str(ex)))
            return None

    attempt("stock_symbol_arg", lambda: vn.stock(symbol="VNM").quote.history(start=s, end=e, interval="1D"))
    attempt("stock_positional", lambda: vn.stock("VNM").quote.history(start=s, end=e, interval="1D"))
    attempt("stock_history_direct", lambda: vn.stock(symbol="VNM").history(start=s, end=e, interval="1D"))
    attempt("vn.history_func", lambda: getattr(vn, "history", lambda *a, **k: None)(symbol="VNM", start=s, end=e, interval="1D"))
    attempt("vn.get_stock", lambda: getattr(vn, "get_stock", lambda *a, **k: None)("VNM", start=s, end=e))
    attempt("vn.stock_chart", lambda: getattr(vn.stock(symbol="VNM"), "chart", lambda *a, **k: None)().history(start=s, end=e))

    try:
        stock_obj = vn.stock(symbol="VNM")
        for attr in dir(stock_obj):
            if "history" in attr.lower() or "quote" in attr.lower() or "chart" in attr.lower():
                nm = f"stock_obj.{attr}"
                def make_fn(a=attr):
                    return lambda: getattr(stock_obj, a)(start=s, end=e) if callable(getattr(stock_obj, a, None)) else getattr(stock_obj, a)
                attempt(nm, make_fn())
    except Exception:
        pass

    return attempts

# --- Phân tích dữ liệu ---
def detect_close_column(df):
    if df is None or len(df) == 0:
        return None
    cols = df.columns.tolist()
    lower = [c.lower() for c in cols]
    candidates = []
    for i, c in enumerate(lower):
        if "close" in c or c in ("matched", "matched_price", "last", "price", "cls"):
            candidates.append(cols[i])
    if not candidates:
        numeric = df.select_dtypes(include=['number']).columns.tolist()
        if numeric:
            return numeric[-1]
        return None
    return candidates[0]

def analyze_df(df, source_name):
    if df is None:
        print(f"{source_name}: no dataframe to analyze")
        return
    close_col = detect_close_column(df)
    print(f"{source_name}: chosen close column = {close_col}")
    if close_col is None:
        return
    series = pd.to_numeric(df[close_col], errors='coerce')
    n_total = len(series)
    n_na = series.isna().sum()
    n_zero = (series == 0).sum()
    n_neg = (series < 0).sum()
    print(f"{source_name}: total={n_total}, NaN={n_na}, zeros={n_zero}, negative={n_neg}")
    if n_total > 0:
        print(f"{source_name}: zero% = {n_zero / n_total * 100:.2f}%  | NaN% = {n_na / n_total * 100:.2f}%")
    print(series.describe())
    return {
        "close_col": close_col,
        "total": n_total,
        "nan": n_na,
        "zero": n_zero,
        "neg": n_neg,
        "series": series
    }

def plot_compare(df_yahoo, info_y, df_vnstock, info_vn):
    try:
        plt.figure(figsize=(12,6))
        # Yahoo
        plt.plot(pd.to_datetime(df_yahoo["Date"]),
                 pd.to_numeric(df_yahoo[info_y["close_col"]], errors="coerce"),
                 label="Yahoo")
        # Vnstock
        plt.plot(pd.to_datetime(df_vnstock["time"]),
                 pd.to_numeric(df_vnstock[info_vn["close_col"]], errors="coerce"),
                 label="Vnstock")

        plt.legend()
        plt.grid(True)
        plt.title("Yahoo vs Vnstock - Close")
        plt.tight_layout()
        plot_path = os.path.join(DATA_DIR, "compare_close.png")
        plt.savefig(plot_path)
        print("Saved plot:", plot_path)
        plt.show()
    except Exception as e:
        print("Plot error:", e)
        traceback.print_exc()


if __name__ == "__main__":
    print("Python:", sys.version)
    print("Pandas:", pd.__version__)
    try:
        import vnstock
        print("vnstock package version:", getattr(vnstock, "__version__", "unknown"))
    except Exception as e:
        print("vnstock import problem:", e)

    # 1) Yahoo data
    df_yahoo = get_yahoo()

    # 2) Vnstock thử nhiều cách
    attempts = try_vnstock_methods()

    # 3) Lấy dataframe usable đầu tiên
    df_vnstock_found = None
    for name, df, err in attempts:
        if df is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
            df_vnstock_found = (name, df)
            break

    if df_vnstock_found is None:
        print("\nNo usable dataframe returned from vnstock attempts. See attempt logs above.")
        sys.exit(1)

    name, df_vnstock = df_vnstock_found
    print(f"\nUsing result from attempt: {name}")

    # 4) Phân tích
    info_vn = analyze_df(df_vnstock, f"Vnstock ({name})")
    info_y = analyze_df(df_yahoo, "Yahoo")

    # 5) Vẽ biểu đồ
    if info_vn and info_y and info_vn["close_col"] and info_y["close_col"]:
        plot_compare(df_yahoo, info_y, df_vnstock, info_vn)
    else:
        print("Data invalid -> not plotting")

