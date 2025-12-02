import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Vẽ 4 biểu đồ cổ phiếu: Close, Volume, OHLC, Rolling stats")
    parser.add_argument("--file", "-f", required=True, help="Đường dẫn tới file CSV")
    parser.add_argument("--datecol", "-d", default="time", help="Tên cột thời gian")
    parser.add_argument("--pricecols", "-p", nargs=4, default=["open","high","low","close"], help="Tên các cột OHLC")
    parser.add_argument("--volcol", "-v", default="volume", help="Tên cột volume")
    parser.add_argument("--rolling_window", "-w", type=int, default=7, help="Window cho rolling mean/std")
    parser.add_argument("--outdir", "-o", default="viz_outputs", help="Thư mục lưu ảnh")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print("❌ Không tìm thấy file CSV!")
        return

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.file)
    df[args.datecol] = pd.to_datetime(df[args.datecol])
    df = df.sort_values(by=args.datecol)

    # 1️⃣ Close Price line chart
    if "close" in df.columns:
        plt.figure(figsize=(12,5))
        plt.plot(df[args.datecol], df["close"], label="Close", color="orange", linewidth=2)
        plt.title("Close Price theo thời gian")
        plt.xlabel("Thời gian")
        plt.ylabel("Giá")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(args.outdir, "close_price.png"))
        plt.close()
        print(f"✅ Lưu Close Price")
    else:
        print("⚠️ Không tìm thấy cột Close, bỏ qua.")

    # 2️⃣ Volume bar chart
    if args.volcol in df.columns:
        plt.figure(figsize=(12,4))
        plt.bar(df[args.datecol], df[args.volcol], color="skyblue")
        plt.title("Volume theo thời gian")
        plt.xlabel("Thời gian")
        plt.ylabel("Volume")
        plt.grid(True)
        plt.savefig(os.path.join(args.outdir, "volume.png"))
        plt.close()
        print(f"✅ Lưu Volume")
    else:
        print("⚠️ Không tìm thấy cột volume, bỏ qua.")

    # 3️⃣ OHLC + High-Low range
    for col in args.pricecols:
        if col not in df.columns:
            print(f"⚠️ Không tìm thấy cột {col}, bỏ qua OHLC.")
            return
    plt.figure(figsize=(12,5))
    plt.fill_between(df[args.datecol], df["low"], df["high"], color="lightblue", alpha=0.3, label="High-Low range")
    plt.plot(df[args.datecol], df["open"], label="Open", color="blue", linewidth=1.5)
    plt.plot(df[args.datecol], df["high"], label="High", color="green", linewidth=1.5)
    plt.plot(df[args.datecol], df["low"], label="Low", color="red", linewidth=1.5)
    plt.plot(df[args.datecol], df["close"], label="Close", color="orange", linewidth=2)
    plt.title("Biểu đồ 4 đặc trưng OHLC + High-Low range")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.outdir, "ohlc_visual.png"))
    plt.close()
    print(f"✅ Lưu OHLC trực quan")

    # 4️⃣ Rolling mean / std
    if "close" in df.columns:
        df["rolling_mean"] = df["close"].rolling(window=args.rolling_window).mean()
        df["rolling_std"] = df["close"].rolling(window=args.rolling_window).std()
        plt.figure(figsize=(12,5))
        plt.plot(df[args.datecol], df["close"], label="Close", color="orange", linewidth=1.5)
        plt.plot(df[args.datecol], df["rolling_mean"], label=f"Rolling Mean ({args.rolling_window})", color="blue", linewidth=1.5)
        plt.fill_between(df[args.datecol],
                         df["rolling_mean"] - df["rolling_std"],
                         df["rolling_mean"] + df["rolling_std"],
                         color="lightgreen", alpha=0.3, label="Rolling Std")
        plt.title(f"Rolling Mean & Std (window={args.rolling_window})")
        plt.xlabel("Thời gian")
        plt.ylabel("Giá")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(args.outdir, "rolling_stats.png"))
        plt.close()
        print(f"✅ Lưu Rolling Mean/Std")
    else:
        print("⚠️ Không tìm thấy cột Close, bỏ qua Rolling stats.")

    print("\n🎉 Hoàn tất. 4 ảnh đã lưu trong:", args.outdir)

if __name__ == "__main__":
    main()
