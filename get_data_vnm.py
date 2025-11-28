import os
import glob
import pandas as pd
from vnstock import Vnstock
from datetime import datetime

# --- 1. DANH SÁCH MÃ CỔ PHIẾU LỚN ---
TICKERS = [
    "VNM",  # Vinamilk
    "VCB",  # Vietcombank
    "CTG",  # VietinBank
    "BID",  # BIDV
    "FPT",  # FPT Corp
    "MWG",  # Mobile World
    "VIC",  # Vingroup
    "VHM",  # Vinhomes
    "HPG",  # Hòa Phát
    "SSI",  # SSI Securities
    "VTP",  # Viettel Post
    "CTR",  # Viettel Construction
]

# --- 2. THƯ MỤC LƯU DỮ LIỆU ---
data_folder = os.path.join(os.path.dirname(__file__), "DATA")
os.makedirs(data_folder, exist_ok=True)

# --- 3. XÓA FILE CSV CŨ ---
for f in glob.glob(os.path.join(data_folder, "*.csv")):
    os.remove(f)
print("✅ Đã xóa các file CSV cũ trong DATA\n")

# --- 4. HÀM TẢI DỮ LIỆU TỪ VNSTOCK ---
def download_vnstock_data(ticker):
    print(f"📥 [Vnstock] Đang tải dữ liệu cho {ticker}...")
    try:
        vn = Vnstock()
        stock = vn.stock(symbol=ticker, source="VCI")  # Có thể đổi sang 'TCBS' nếu cần
        df = stock.quote.history(
            start="2005-01-01",
            end=datetime.now().strftime("%Y-%m-%d"),
            interval="1D"
        )
        if df is None or df.empty:
            print(f"  ⚠️ Không có dữ liệu từ Vnstock cho {ticker}")
            return None

        df = df.rename(columns=str.lower)
        df["symbol"] = ticker
        print(f"  ✅ Hoàn tất: {len(df)} dòng dữ liệu.")
        return df

    except Exception as e:
        print(f"  ❌ Lỗi khi tải {ticker}: {e}")
        return None

# --- 5. HÀM CHÍNH ---
summary = []
for ticker in TICKERS:
    df = download_vnstock_data(ticker)
    if df is not None and not df.empty:
        path = os.path.join(data_folder, f"{ticker}_data.csv")
        df.to_csv(path, index=False)
        summary.append({"Ticker": ticker, "Rows": len(df), "Source": "Vnstock"})
        print(f"📄 Đã lưu {ticker} ({len(df)} dòng)\n")
    else:
        summary.append({"Ticker": ticker, "Rows": 0, "Source": "None"})
        print(f"❌ Bỏ qua {ticker} do không có dữ liệu.\n")

# --- 6. GHI FILE TỔNG HỢP ---
summary_df = pd.DataFrame(summary)
summary_path = os.path.join(data_folder, "data_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\n🎯 Hoàn tất tải dữ liệu từ Vnstock!")
print(summary_df)
print(f"\n📄 Log lưu tại: {summary_path}")
