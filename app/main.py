import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import MinMaxScaler
from .train_model import train_lstm
from .analysis import analyze_monthly, analyze_quarterly
from .signals import signal_recommend, signal_simple, signal_advanced, signal_summary
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "MODELS")
DATA_DIR = os.path.join(PROJECT_ROOT, "DATA")
PRED_DIR = os.path.join(PROJECT_ROOT, "DATA_PREDICT")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app = FastAPI(title="Stock Prediction API")
def model_file_path(file_name: str):
    base = os.path.basename(file_name).replace(".csv", "_lstm.h5")
    return os.path.join(MODELS_DIR, base)
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# STATIC
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/predictions", StaticFiles(directory=PRED_DIR), name="predictions")


def pred_file_name(file):
    return file.replace(".csv", "_du_doan.csv")


def pred_file_path(file):
    return os.path.join(PRED_DIR, pred_file_name(file))


# ============================
# API LIST FILES
# ============================
@app.get("/files")
def list_files():
    try:
        files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
        return {"ok": True, "files": files}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


# ============================
# UPLOAD CSV
# ============================
@app.post("/upload")
async def upload_csv(file: UploadFile):
    try:
        save_path = os.path.join(DATA_DIR, file.filename)
        df = pd.read_csv(file.file)
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        return {"ok": True, "msg": "Upload thành công", "file": file.filename}
    except Exception as e:
        return {"ok": False, "msg": str(e)}

# ============================
# TRAIN MODEL
# ============================
@app.post("/train")
async def train(file: str = Form(...)):
    try:
        data_path = os.path.join(DATA_DIR, file)
        out_path, mae, rmse, model_path = train_lstm(data_path)
        return {
            "ok": True,
            "msg": "Train thành công",
            "predict_file": os.path.basename(out_path),
            "model_file": os.path.basename(model_path),
            "mae": mae,
            "rmse": rmse,
        }
    except Exception as e:
        return {"ok": False, "msg": str(e)}


# CHECK PREDICT EXISTS
@app.get("/predict/check")
def check_predict(file: str):
    return {"exists": os.path.exists(pred_file_path(file))}


# ============================
# FIXED API /predict/data
# ============================
@app.get("/predict/data")
def predict_data(file: str, range: int = 365):

    pred_path = pred_file_path(file)
    if not os.path.exists(pred_path):
        return JSONResponse({"ok": False, "msg": "Chưa có dữ liệu dự đoán"}, status_code=400)

    df = pd.read_csv(pred_path)

    # Convert date
    df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce")
    df = df.dropna(subset=["Ngày"])

    # Limit range
    if range > 0:
        df = df.tail(range)

    return {
        "ok": True,
        "dates": df["Ngày"].dt.strftime("%Y-%m-%d").tolist(),
        "real": df["Giá_đóng_cửa_thực_tế"].astype(float).tolist(),
        "pred": df["Giá_đóng_cửa_dự_đoán"].astype(float).tolist()
    }


# ============================
# MONTHLY
# ============================
@app.get("/predict/monthly")
def monthly(file: str):
    pred_path = pred_file_path(file)
    data, err = analyze_monthly(pred_path)
    if err:
        return {"ok": False, "msg": err}
    return {"ok": True, "data": data}


# ============================
# QUARTERLY
# ============================
@app.get("/predict/quarterly")
def quarterly(file: str):
    pred_path = pred_file_path(file)
    data, err = analyze_quarterly(pred_path)
    if err:
        return {"ok": False, "msg": err}
    return {"ok": True, "data": data}


# ============================
# DASHBOARD
# ============================
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard_pro.html", {"request": request})


# ============================
# SIGNALS
# ============================
@app.post("/signal/simple")
def api_signal_simple(file: str = Form(...), horizon: int = Form(7)):
    pred_path = pred_file_path(file)
    return signal_simple(pred_path, horizon)
@app.post("/signal/advanced")
def api_signal_advanced(file: str = Form(...)):
    """
    Nhận tên file từ FormData, trả về signal luôn hợp lệ.
    """
    try:
        pred_path = pred_file_path(file)
        result = signal_advanced(pred_path)
        # Bảo vệ tuyệt đối signal
        if result.get("signal") not in ["STRONG_BUY","BUY","HOLD","SELL","STRONG_SELL"]:
            result["signal"] = "HOLD"
        return result
    except Exception as e:
        return {"ok": False, "signal": "HOLD", "score": 0.0, "note": f"Lỗi server: {str(e)}"}
@app.post("/signal/summary")
def api_signal_summary(file: str = Form(...)):
    return signal_summary(PROJECT_ROOT, file)
@app.post("/signal/recommend")
def api_signal_recommend(file: str = Form(...)):
    pred_path = pred_file_path(file)
    return signal_recommend(pred_path)
@app.get("/predict/next-month")
def predict_next_month(file: str):
    # Kiểm tra file dự đoán
    pred_path = pred_file_path(file)
    if not os.path.exists(pred_path):
        return {"ok": False, "msg": "Chưa có file dự đoán"}

    # Kiểm tra file model
    model_path = model_file_path(file)
    if not os.path.exists(model_path):
        return {"ok": False, "msg": "Chưa có file model"}

    # Load dữ liệu dự đoán
    df = pd.read_csv(pred_path)
    df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce")
    df = df.dropna().sort_values("Ngày")

    # Load model LSTM
    model = load_model(model_path)

    # Chọn đúng 5 features đã dùng lúc train
    feature_cols = ["open", "high", "low", "close", "volume"]

    # Nếu file predict không có đủ cột, fallback dùng giá close lặp lại
    if not all(col in df.columns for col in feature_cols):
        # chuyển các cột giả lập từ close
        for i, col in enumerate(feature_cols):
            if col not in df.columns:
                df[col] = df["Giá_đóng_cửa_dự_đoán"] if "Giá_đóng_cửa_dự_đoán" in df.columns else df["close"]

    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols].values)

    window_size = 60
    last_sequence = scaled[-window_size:]
    last_sequence = last_sequence.reshape(1, window_size, len(feature_cols))

    # Dự báo 30 ngày tiếp theo
    next_month_scaled = []
    current_sequence = last_sequence.copy()
    for _ in range(30):
        pred_scaled = model.predict(current_sequence, verbose=0)  # shape=(1,4)
        next_month_scaled.append(pred_scaled[0,3])  # cột close

        # cập nhật sequence: bỏ dòng đầu, thêm giá dự đoán
        new_row = current_sequence[:, -1, :].copy()
        new_row[0,3] = pred_scaled[0,3]  # chỉ cập nhật close
        current_sequence = np.append(current_sequence[:,1:,:], new_row.reshape(1,1,len(feature_cols)), axis=1)

    # Chuyển về giá gốc
    # scaler đã fit 5 features → tạo array đủ 5 cột, chỉ set cột close mới
    next_month_array = np.zeros((30, len(feature_cols)))
    next_month_array[:,3] = next_month_scaled
    next_month = scaler.inverse_transform(next_month_array)[:,3]
    next_month = [float(p) for p in next_month]

    return {"ok": True, "next_month": next_month}
@app.get("/predict/next-quarter")
def predict_next_quarter(file: str):
    # Kiểm tra file dự đoán
    pred_path = pred_file_path(file)
    if not os.path.exists(pred_path):
        return {"ok": False, "msg": "Chưa có file dự đoán"}

    # Kiểm tra file model
    model_path = model_file_path(file)
    if not os.path.exists(model_path):
        return {"ok": False, "msg": "Chưa có file model"}

    # Load dữ liệu dự đoán
    df = pd.read_csv(pred_path)
    df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce")
    df = df.dropna().sort_values("Ngày")

    # Load model LSTM
    model = load_model(model_path)

    # Chọn đúng 5 features
    feature_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in feature_cols):
        for i, col in enumerate(feature_cols):
            if col not in df.columns:
                df[col] = df["Giá_đóng_cửa_dự_đoán"] if "Giá_đóng_cửa_dự_đoán" in df.columns else df["close"]

    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols].values)

    window_size = 60
    last_sequence = scaled[-window_size:]
    last_sequence = last_sequence.reshape(1, window_size, len(feature_cols))

    # Dự báo 90 ngày tiếp theo
    next_quarter_scaled = []
    current_sequence = last_sequence.copy()
    for _ in range(90):
        pred_scaled = model.predict(current_sequence, verbose=0)  # shape=(1,4)
        next_quarter_scaled.append(pred_scaled[0,3])  # chỉ lấy cột close

        # cập nhật sequence: bỏ dòng đầu, thêm giá dự đoán
        new_row = current_sequence[:, -1, :].copy()
        new_row[0,3] = pred_scaled[0,3]
        current_sequence = np.append(current_sequence[:,1:,:], new_row.reshape(1,1,len(feature_cols)), axis=1)

    # Chuyển về giá gốc
    next_quarter_array = np.zeros((90, len(feature_cols)))
    next_quarter_array[:,3] = next_quarter_scaled
    next_quarter = scaler.inverse_transform(next_quarter_array)[:,3]
    next_quarter = [float(p) for p in next_quarter]

    return {"ok": True, "next_quarter": next_quarter}

@app.get("/signal/recommend")
def recommend(file: str):
    return signal_summary(PROJECT_ROOT, file)
