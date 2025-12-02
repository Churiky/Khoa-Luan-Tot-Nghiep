import os
import math
from xml.parsers.expat import model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.tseries.offsets import DateOffset

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "DATA")
PREDICT_DIR = os.path.join(BASE_DIR, "DATA_PREDICT")
os.makedirs(PREDICT_DIR, exist_ok=True)

N_STEPS = 60
TEST_RATIO = 0.15
NOISE_STD = 1e-3


# ==========================================================
# AUTO FIX FORMAT CSV
# ==========================================================
def detect_and_fix_format(df):
    cols = {c.lower().strip(): c for c in df.columns}
    rename_map = {}

    for k in ["date", "time", "ngày", "day"]:
        if k in cols:
            rename_map[cols[k]] = "date"

    for k in ["close", "c", "giá", "giá_đóng_cửa", "price"]:
        if k in cols:
            rename_map[cols[k]] = "close"

    for k in ["open", "o"]:
        if k in cols:
            rename_map[cols[k]] = "open"

    for k in ["high", "h"]:
        if k in cols:
            rename_map[cols[k]] = "high"

    for k in ["low", "l"]:
        if k in cols:
            rename_map[cols[k]] = "low"

    for k in ["volume", "vol", "kl", "giao_dịch"]:
        if k in cols:
            rename_map[cols[k]] = "volume"

    df = df.rename(columns=rename_map)

    if "date" not in df.columns:
        raise ValueError("File CSV cần có cột ngày.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["date"]).sort_values("date")

    if "close" not in df.columns:
        raise ValueError("File CSV cần có cột giá (close/giá/price).")

    if "open" not in df.columns:
        df["open"] = df["close"].shift(1).fillna(df["close"])

    if "high" not in df.columns:
        df["high"] = df["close"] * 1.002

    if "low" not in df.columns:
        df["low"] = df["close"] * 0.998

    if "volume" not in df.columns:
        df["volume"] = 0

    return df[["date", "open", "high", "low", "close", "volume"]]

def inverse_column(scaled_col, scaler, col_index):
    """
    scaled_col: dữ liệu đã scale (1 cột)
    scaler: MinMaxScaler đã fit
    col_index: vị trí cột trong scaler.fit (open=0, high=1, low=2, close=3, volume=4)
    """
    dummy = np.zeros((len(scaled_col), 5))
    dummy[:, col_index] = scaled_col
    return scaler.inverse_transform(dummy)[:, col_index]
# ==========================================================
# TRAIN MODEL — LAZY IMPORT (fix crash)
# ==========================================================
def train_lstm(file_path, progress_callback=None, epochs=80, batch_size=16, future_steps=365):

    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.callbacks import Callback, EarlyStopping
    from keras.losses import Huber

    def pb(p, msg=""):
        if progress_callback:
            try:
                progress_callback(int(p), msg)
            except:
                pass

    pb(5, "Đang đọc dữ liệu...")

    raw_df = pd.read_csv(file_path)
    raw_df.columns = [c.strip() for c in raw_df.columns]

    # CASE: file dự đoán tương lai (chỉ có 2 cột)
    if len(raw_df.columns) == 2:
        date_col = raw_df.columns[0]
        price_col = raw_df.columns[1]

        df = raw_df.copy()
        df["Giá_đóng_cửa_dự_đoán"] = df[price_col].astype(float)

        future_path = file_path
        merged_data = {
            "dates": df[date_col].tolist(),
            "real": [None] * len(df),
            "pred": df["Giá_đóng_cửa_dự_đoán"].tolist()
        }
        return None, future_path, None, None, None, merged_data
    df = detect_and_fix_format(raw_df)

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df[df["close"] > 0]
    if df.empty:
        raise ValueError("Không còn dữ liệu hợp lệ sau khi lọc NaN hoặc close <= 0")
    df = df.sort_values("date") 

    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)

    if len(df) < N_STEPS + 1:
        raise ValueError(f"Dữ liệu quá ít (>= {N_STEPS + 1}).")

    pb(10, "Chuẩn hóa dữ liệu...")

    value_cols = ["open", "high", "low", "close", "volume"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[value_cols])

    split_idx = int(len(df) * (1 - TEST_RATIO))
    train_scaled = scaled[:split_idx]
    test_scaled = scaled[split_idx:]

    train_scaled += np.random.normal(0, NOISE_STD, train_scaled.shape)

    # ==========================================
    # TẠO X,Y CHỈ DỰ ĐOÁN CLOSE!!!
    # ==========================================
    X_train, y_train = [], []
    for i in range(N_STEPS, len(train_scaled)):
        X_train.append(train_scaled[i - N_STEPS:i])
        y_train.append(train_scaled[i, 3])  # Chỉ lấy cột close
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # test
    full_scaled = np.concatenate([train_scaled, test_scaled])
    X_test, y_test = [], []
    for i in range(len(train_scaled), len(full_scaled)):
        X_test.append(full_scaled[i - N_STEPS:i])
        y_test.append(full_scaled[i, 3])
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    pb(25, "Xây dựng mô hình LSTM...")

    # =======================================
    # MÔ HÌNH ĐƠN GIẢN + CHỈ RA CLOSE
    # =======================================
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),

        LSTM(32),
        Dropout(0.2),

        Dense(1)  # Chỉ dự đoán 1 giá close
    ])

    model.compile(optimizer="nadam", loss=Huber(delta=1.0))
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    class PB(Callback):
        def on_epoch_end(self, epoch, logs=None):
            msg = f"[TRAIN] Epoch {epoch+1} - loss={logs.get('loss'):.6f}"
            pb(30 + int(min(45, (epoch + 1) * 0.7)), msg)

    pb(30, "Bắt đầu training...")

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[PB(), early_stop]
    )

    pb(80, "Dự đoán dữ liệu...")

    pred_scaled = model.predict(X_test).reshape(-1)
    last_seq = full_scaled[-N_STEPS:]  # lấy N_STEPS cuối
    future_preds_scaled = []
    curr_seq = last_seq.copy()

    for _ in range(future_steps):
        pred = model.predict(curr_seq[np.newaxis, :, :])[0, 0]
        future_preds_scaled.append(pred)
        next_row = curr_seq[-1].copy()
        next_row[3] = pred  # chỉ update close
        curr_seq = np.vstack([curr_seq[1:], next_row])

    future_pred = inverse_column(np.array(future_preds_scaled), scaler, col_index=3)

    # tạo ngày dự đoán tương ứng từng ngày
    last_date = df.index[-1]
    future_dates = [last_date + DateOffset(days=i+1) for i in range(future_steps)]
    # =====================================================
    # INVERSE TRANSFORM CHUẨN
    # =====================================================
    inv_pred = inverse_column(pred_scaled, scaler, col_index=3)
    inv_true = inverse_column(np.array(y_test), scaler, col_index=3)

    out_len = min(len(inv_pred), len(df.index))

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae = float(mean_absolute_error(inv_true[:out_len], inv_pred[:out_len]))
    rmse = float(math.sqrt(mean_squared_error(inv_true[:out_len], inv_pred[:out_len])))

    pb(95, "Xuất file...")

    out_df = pd.DataFrame({
    "Ngày": df.index[-out_len:].strftime("%Y-%m-%d"),
    "Giá_đóng_cửa_thực_tế": inv_true[-out_len:].astype(float),
    "Giá_đóng_cửa_dự_đoán": inv_pred[-out_len:].astype(float)
})

    out_path = os.path.join(PREDICT_DIR, os.path.basename(file_path).replace(".csv", "_du_doan.csv"))
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # file dự đoán 12 tháng tương lai
    future_df = pd.DataFrame({
        "Ngày": [d.strftime("%Y-%m-%d") for d in future_dates],
        "Giá_đóng_cửa_dự_đoán": future_pred
    })
    future_path = os.path.join(PREDICT_DIR, os.path.basename(file_path).replace(".csv", "_du_doan_tuong_lai.csv"))
    future_df.to_csv(future_path, index=False, encoding="utf-8-sig")

    MODELS_DIR = os.path.join(BASE_DIR, "MODELS")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, os.path.basename(file_path).replace(".csv", "_lstm.h5"))
    model.save(model_path)
    merged_data = {
    "dates": list(df.index[-out_len:].strftime("%Y-%m-%d")) + [d.strftime("%Y-%m-%d") for d in future_dates],
    "real": list(inv_true[-out_len:].astype(float)) + [None]*12,  # future chưa có giá thực tế
    "pred": list(inv_pred[-out_len:].astype(float)) + list(future_pred)
}
    pb(100, "Hoàn tất!")
    

    return out_path, future_path, mae, rmse, model_path, merged_data
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dự báo giá cổ phiếu bằng LSTM")
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Đường dẫn file CSV dữ liệu cổ phiếu"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=80,
        help="Số epoch để huấn luyện (mặc định 80)"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=16,
        help="Kích thước batch (mặc định 16)"
    )
    args = parser.parse_args()

    # Hàm progress đơn giản in ra console
    def progress(percent, msg=""):
        print(f"[{percent}%] {msg}")

    try:
        out_path, future_path, mae, rmse, model_path, merged_data = train_lstm(
    args.file,
    progress_callback=progress,
    epochs=args.epochs,
    batch_size=args.batch_size
)
        print(f"\nHoàn tất! File dự đoán: {out_path}")
        print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    except Exception as e:
        print("Có lỗi xảy ra:", e)