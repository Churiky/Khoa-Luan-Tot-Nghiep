import os
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
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


# ==========================================================
# TRAIN MODEL — LAZY IMPORT (fix crash)
# ==========================================================
def train_lstm(file_path, progress_callback=None, epochs=80, batch_size=16):

    # ============================
    # IMPORT TENSORFLOW TẠI ĐÂY !!!
    # ============================
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, Bidirectional
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
    df = detect_and_fix_format(raw_df)

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df[df["close"] > 0]
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

    X_train, y_train = [], []
    for i in range(N_STEPS, len(train_scaled)):
        X_train.append(train_scaled[i - N_STEPS:i])
        y_train.append(train_scaled[i, :4])
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    full_scaled = np.concatenate([train_scaled, test_scaled])
    X_test, y_test = [], []
    for i in range(len(train_scaled), len(full_scaled)):
        X_test.append(full_scaled[i - N_STEPS:i])
        y_test.append(full_scaled[i, :4])
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    pb(25, "Xây dựng mô hình LSTM...")

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(4)
    ])  

    model.compile(optimizer="nadam", loss=Huber(delta=1.0))
    early_stop = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)

    class PB(Callback):
        def on_epoch_end(self, epoch, logs=None):
            msg = f"[TRAIN] Epoch {epoch+1} - loss={logs.get('loss'):.6f}"
            print(msg)
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

    pred_scaled = model.predict(X_test)

    inv_pred, inv_true = [], []
    for i in range(4):
        inv_pred.append(scaler.inverse_transform(
            np.concatenate([pred_scaled[:, i:i+1], np.zeros((len(pred_scaled), 4))], axis=1)
        )[:, 0])

        inv_true.append(scaler.inverse_transform(
            np.concatenate([y_test[:, i:i+1], np.zeros((len(y_test), 4))], axis=1)
        )[:, 0])

    inv_pred = np.array(inv_pred).T
    inv_true = np.array(inv_true).T

    out_len = min(len(inv_pred), len(df.index))

    mae = float(mean_absolute_error(inv_true[:out_len, 3], inv_pred[:out_len, 3]))
    rmse = float(math.sqrt(mean_squared_error(inv_true[:out_len, 3], inv_pred[:out_len, 3])))

    pb(95, "Xuất file...")

    out_df = pd.DataFrame({
        "Ngày": df.index[-out_len:].strftime("%Y-%m-%d"),
        "Giá_đóng_cửa_thực_tế": inv_true[:out_len, 3].astype(float),
        "Giá_đóng_cửa_dự_đoán": inv_pred[:out_len, 3].astype(float)
    })

    out_path = os.path.join(PREDICT_DIR, os.path.basename(file_path).replace(".csv", "_du_doan.csv"))
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    MODELS_DIR = os.path.join(BASE_DIR, "MODELS")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, os.path.basename(file_path).replace(".csv", "_lstm.h5"))
    model.save(model_path)
    pb(98, f"Lưu model LSTM: {model_path}")

    pb(100, "Hoàn tất!")

    return out_path, mae, rmse,model_path
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
        out_path, mae, rmse, model_path = train_lstm(
    args.file,
    progress_callback=progress,
    epochs=args.epochs,
    batch_size=args.batch_size
)
        print(f"\nHoàn tất! File dự đoán: {out_path}")
        print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    except Exception as e:
        print("Có lỗi xảy ra:", e)