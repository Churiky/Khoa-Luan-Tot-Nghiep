import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.losses import Huber
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# ==========================
# Cấu hình
# ==========================
DATA_FILE = "DATA/VTP_data.csv"  # 🔹 Đổi thành file của bạn
N_STEPS = 60
TEST_RATIO = 0.15
EPOCHS = 60
BATCH_SIZE = 16

# ==========================
# Đọc và xử lý dữ liệu
# ==========================
df = pd.read_csv(DATA_FILE)
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df = df.dropna(subset=['time'])
df = df.sort_values('time').reset_index(drop=True)
df.rename(columns={'time': 'date'}, inplace=True)

value_cols = ['open', 'high', 'low', 'close', 'volume']
df[value_cols] = df[value_cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=value_cols)
df.set_index('date', inplace=True)

# Chia tập train/test
split_idx = int(len(df) * (1 - TEST_RATIO))
train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

scalers = {col: MinMaxScaler((0, 1)) for col in value_cols}

def scale_data(dataframe):
    scaled = np.zeros_like(dataframe.values)
    for i, col in enumerate(value_cols):
        scaled[:, i] = scalers[col].fit_transform(dataframe[[col]]).ravel()
    return scaled

scaled_train = scale_data(train_df)
scaled_test = scale_data(test_df)

def create_xy(data_scaled):
    X, y = [], []
    for i in range(N_STEPS, len(data_scaled)):
        X.append(data_scaled[i-N_STEPS:i])
        y.append(data_scaled[i, :4])
    return np.array(X), np.array(y)

X_train, y_train = create_xy(scaled_train)
X_test, y_test = create_xy(np.concatenate([scaled_train, scaled_test]))

# ==========================
# Hàm xây dựng mô hình
# ==========================
def build_model(loss_fn):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(N_STEPS, 5)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(4)
    ])
    model.compile(optimizer='adam', loss=loss_fn)
    return model

# ==========================
# Huấn luyện 2 mô hình
# ==========================
model_mse = build_model('mse')
model_huber = build_model(Huber(delta=1.0))

print("🔹 Huấn luyện với MSE...")
hist_mse = model_mse.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                         validation_data=(X_test, y_test), verbose=0)

print("🔹 Huấn luyện với Huber Loss...")
hist_huber = model_huber.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                             validation_data=(X_test, y_test), verbose=0)

# ==========================
# Dự đoán và đánh giá
# ==========================
def evaluate_model(model, name):
    pred_scaled = model.predict(X_test)
    inv_pred, inv_true = [], []
    for i, col in enumerate(['open', 'high', 'low', 'close']):
        inv_pred.append(scalers[col].inverse_transform(pred_scaled[:, i].reshape(-1, 1)).ravel())
        inv_true.append(scalers[col].inverse_transform(y_test[:, i].reshape(-1, 1)).ravel())
    inv_pred, inv_true = np.array(inv_pred).T, np.array(inv_true).T
    rmse = math.sqrt(mean_squared_error(inv_true[:, 3], inv_pred[:, 3]))
    mae = mean_absolute_error(inv_true[:, 3], inv_pred[:, 3])
    print(f"✅ {name} — MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return inv_true[:, 3], inv_pred[:, 3]

true_close_mse, pred_close_mse = evaluate_model(model_mse, "MSE")
true_close_huber, pred_close_huber = evaluate_model(model_huber, "Huber")

# ==========================
# Vẽ biểu đồ so sánh
# ==========================
plt.figure(figsize=(14, 5))
plt.plot(hist_mse.history['loss'], label='Train Loss (MSE)')
plt.plot(hist_huber.history['loss'], label='Train Loss (Huber)')
plt.plot(hist_mse.history['val_loss'], '--', label='Val Loss (MSE)')
plt.plot(hist_huber.history['val_loss'], '--', label='Val Loss (Huber)')
plt.title('So sánh Loss giữa MSE và Huber')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(true_close_mse[-200:], label='Giá thực tế', color='black')
plt.plot(pred_close_mse[-200:], label='Dự đoán (MSE)', alpha=0.7)
plt.plot(pred_close_huber[-200:], label='Dự đoán (Huber)', alpha=0.7)
plt.title('So sánh dự đoán giá đóng cửa - MSE vs Huber')
plt.xlabel('Ngày')
plt.ylabel('Giá đóng cửa')
plt.legend()
plt.grid()
plt.show()
