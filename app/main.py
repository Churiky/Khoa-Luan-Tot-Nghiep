import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from .train_model import train_lstm
from .analysis import analyze_monthly, analyze_quarterly
from .signals import signal_simple, signal_advanced, signal_summary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "DATA")
PRED_DIR = os.path.join(PROJECT_ROOT, "DATA_PREDICT")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app = FastAPI(title="Stock Prediction API")

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
        out, mae, rmse = train_lstm(data_path)
        return {
            "ok": True,
            "msg": "Train thành công",
            "predict_file": os.path.basename(out),
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
    pred_path = pred_file_path(file)
    return signal_advanced(pred_path)


@app.post("/signal/summary")
def api_signal_summary(file: str = Form(...)):
    return signal_summary(PROJECT_ROOT, file)

@app.get("/predict/next-month")
def predict_next_month(file: str):

    pred_path = pred_file_path(file)
    if not os.path.exists(pred_path):
        return {"ok": False, "msg": "Chưa có file dự đoán"}

    df = pd.read_csv(pred_path)
    df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce")
    df = df.dropna().sort_values("Ngày")

    last_price = df["Giá_đóng_cửa_dự_đoán"].iloc[-1]

    next_month = []
    p = last_price
    for i in range(30):
        p = p * (1 + np.random.normal(0.0015, 0.003))
        next_month.append(float(p))

    return {"ok": True, "next_month": next_month}

@app.get("/predict/next-quarter")
def predict_next_quarter(file: str):

    pred_path = pred_file_path(file)
    if not os.path.exists(pred_path):
        return {"ok": False, "msg": "Chưa có file dự đoán"}

    df = pd.read_csv(pred_path)
    df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce")
    df = df.dropna().sort_values("Ngày")

    last_price = df["Giá_đóng_cửa_dự_đoán"].iloc[-1]

    next_quarter = []
    p = last_price
    for i in range(90):
        p = p * (1 + np.random.normal(0.001, 0.002))
        next_quarter.append(float(p))

    return {"ok": True, "next_quarter": next_quarter}

@app.get("/signal/recommend")
def recommend(file: str):
    return signal_summary(PROJECT_ROOT, file)
