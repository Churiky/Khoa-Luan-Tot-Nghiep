# ============================================================
#                      IMPORTS
# ============================================================
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import uuid

# MODULE của bạn
from .train_model import train_lstm
from .analysis import analyze_monthly, analyze_quarterly
from .signals import signal_recommend, signal_simple, signal_advanced, signal_summary
from .get_data_api import download_stock


# ============================================================
#                      PATH SETUP
# ============================================================
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
    return os.path.join(MODELS_DIR, file_name.replace(".csv", "_lstm.h5"))


def pred_file_name(file):
    return file.replace(".csv", "_du_doan.csv")


def pred_file_path(file):
    return os.path.join(PRED_DIR, pred_file_name(file))


# ============================================================
#                      CORS
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
#                      STATIC ROUTES
# ============================================================
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/predictions", StaticFiles(directory=PRED_DIR), name="predictions")


# ============================================================
#                      DOWNLOAD CSV FROM API
# ============================================================
@app.post("/api/download")
async def download_symbol(symbol: str = Form(...)):
    df, source = download_stock(symbol)
    if df is None:
        return {"ok": False, "msg": "Không thể lấy dữ liệu từ API!"}

    save_path = os.path.join(DATA_DIR, f"{symbol.upper()}.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    return {"ok": True, "file": f"{symbol.upper()}.csv", "source": source}


# ============================================================
#                      CSV LIST
# ============================================================
@app.get("/files")
def list_files():
    try:
        files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
        return {"ok": True, "files": files}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


# ============================================================
#                      UPLOAD CSV
# ============================================================
@app.post("/upload")
async def upload_csv(file: UploadFile):
    try:
        save_path = os.path.join(DATA_DIR, file.filename)
        df = pd.read_csv(file.file)
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        return {"ok": True, "msg": "Upload thành công", "file": file.filename}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


# ============================================================
#                      TRAIN MODEL
# ============================================================
@app.post("/train")
async def train(file: str = Form(...), force: int = Form(0)):
    data_path = os.path.join(DATA_DIR, file)
    pred_path = pred_file_path(file)
    model_path = model_file_path(file)

    if os.path.exists(pred_path) and os.path.exists(model_path) and force == 0:
        return {
            "ok": True,
            "need_confirm": True,
            "msg": f"⛔ File '{file}' đã được train.",
            "predict_file": os.path.basename(pred_path),
            "model_file": os.path.basename(model_path)
        }

    try:
        out_path, future_path, mae, rmse, mpath, merged_data = train_lstm(data_path)
        return {
            "ok": True,
            "msg": "Train thành công",  
            "predict_file": os.path.basename(out_path),
            "model_file": os.path.basename(mpath),
            "mae": mae,
            "rmse": rmse,
        }
    except Exception as e:
        return {"ok": False, "msg": str(e)}


# ============================================================
#             UTILITY: SAFE JSON FOR FLOATS (NO NAN)
# ============================================================
def safe_float_list(col):
    arr = []
    for v in col:
        if pd.isna(v) or np.isinf(v):
            arr.append(None)
        else:
            arr.append(float(v))
    return arr


# ============================================================
#                   PREDICT DATA (FULL + FUTURE)
# ============================================================
@app.get("/predict/check")
def check_predict(file: str):
    return {"exists": os.path.exists(pred_file_path(file))}


@app.get("/predict/data")
def predict_data(file: str, range_days: int = 365, start_date: str = None, end_date: str = None):

    if start_date or end_date:
        pred_file = os.path.join(PRED_DIR, file.replace(".csv", "_du_doan_tuong_lai.csv"))
        if not os.path.exists(pred_file):
            return {"ok": False, "msg": "Chưa có dữ liệu dự đoán tương lai"}

        df = pd.read_csv(pred_file)
        df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce")
        df = df.dropna(subset=["Ngày"]).sort_values("Ngày").reset_index(drop=True)

        if start_date:
            df = df[df["Ngày"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["Ngày"] <= pd.to_datetime(end_date)]

        if df.empty:
            return {"ok": False, "msg": "Không có dữ liệu trong khoảng thời gian này"}

        return {
            "ok": True,
            "dates": df["Ngày"].dt.strftime("%Y-%m-%d").tolist(),
            "real": safe_float_list(df.get("Giá_đóng_cửa_thực_tế", df.get("Giá_đóng_cửa_dự_đoán", []))),
            "pred": safe_float_list(df["Giá_đóng_cửa_dự_đoán"]),
        }

    # Trường hợp user không nhập date → trả dữ liệu bình thường để vẽ biểu đồ
    pred_path = pred_file_path(file)
    if not os.path.exists(pred_path):
        return JSONResponse({"ok": False, "msg": "Chưa có dữ liệu dự đoán"}, status_code=400)

    df = pd.read_csv(pred_path)
    df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce")
    df = df.dropna(subset=["Ngày"]).sort_values("Ngày").reset_index(drop=True)

    df = df.tail(range_days)
    return {
        "ok": True,
        "dates": df["Ngày"].dt.strftime("%Y-%m-%d").tolist(),
        "real": safe_float_list(df.get("Giá_đóng_cửa_thực_tế", df.get("Giá_đóng_cửa_dự_đoán", []))),
        "pred": safe_float_list(df["Giá_đóng_cửa_dự_đoán"]),
    }

# ============================================================
#                 FILTER PRED FILE FOR MONTHLY/QUARTERLY
# ============================================================
def filter_pred_file(pred_path, start_date=None, end_date=None):

    if not start_date and not end_date:
        return pred_path, False

    df = pd.read_csv(pred_path)
    df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce")

    if start_date:
        df = df[df["Ngày"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Ngày"] <= pd.to_datetime(end_date)]

    tmp = os.path.join(PRED_DIR, f"tmp_{uuid.uuid4().hex}.csv")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    return tmp, True


# ============================================================
#                     MONTHLY / QUARTERLY
# ============================================================
@app.get("/predict/monthly")
def monthly(file: str, start_date: str = None, end_date: str = None):

    pred_path = pred_file_path(file)
    used, temp = filter_pred_file(pred_path, start_date, end_date)

    try:
        data, err = analyze_monthly(used)
        if err: return {"ok": False, "msg": err}
        return {"ok": True, "data": data}
    finally:
        if temp and os.path.exists(used):
            os.remove(used)


@app.get("/predict/quarterly")
def quarterly(file: str, start_date: str = None, end_date: str = None):

    pred_path = pred_file_path(file)
    used, temp = filter_pred_file(pred_path, start_date, end_date)

    try:
        data, err = analyze_quarterly(used)
        if err: return {"ok": False, "msg": err}
        return {"ok": True, "data": data}
    finally:
        if temp and os.path.exists(used):
            os.remove(used)


# ============================================================
#                     DASHBOARD RENDER
# ============================================================
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard_pro.html", {"request": request})


# ============================================================
#                     SIGNALS
# ============================================================
@app.post("/signal/simple")
def api_signal_simple(file: str = Form(...), horizon: int = Form(7)):
    return signal_simple(pred_file_path(file), horizon)


@app.post("/signal/advanced")
def api_signal_advanced(
    file: str = Form(...),
    start_date: str = Form(None),
    end_date: str = Form(None)
):
    pred_path = pred_file_path(file)

    # 🔥 lọc file theo khoảng ngày user chọn
    used, temp = filter_pred_file(pred_path, start_date, end_date)

    try:
        result = signal_advanced(used)
        if result.get("signal") not in ["STRONG_BUY", "BUY", "SELL", "STRONG_SELL"]:
            result["signal"] = "HOLD"
        return result
    finally:
        if temp and os.path.exists(used):
            os.remove(used)


@app.post("/signal/summary")
def api_signal_summary(
    file: str = Form(...),
    start_date: str = Form(None),
    end_date: str = Form(None)
):
    pred_path = pred_file_path(file)
    used, temp = filter_pred_file(pred_path, start_date, end_date)

    try:
        # Lưu ý: summary nhận file name → gửi file filtered
        return signal_summary(PROJECT_ROOT, os.path.basename(used))
    finally:
        if temp and os.path.exists(used):
            os.remove(used)


@app.post("/signal/recommend")
def api_signal_recommend(
    file: str = Form(...),
    start_date: str = Form(None),
    end_date: str = Form(None)
):
    pred_path = pred_file_path(file)
    used, temp = filter_pred_file(pred_path, start_date, end_date)

    try:
        return signal_recommend(used)
    finally:
        if temp and os.path.exists(used):
            os.remove(used)


# ============================================================
#             NEXT MONTH / NEXT QUARTER (GIỮ NGUYÊN)
# ============================================================
@app.get("/predict/next-month")
def predict_next_month(file: str):

    pred_path = pred_file_path(file)
    model_path = model_file_path(file)

    if not os.path.exists(pred_path) or not os.path.exists(model_path):
        return {"ok": False, "msg": "Thiếu file dự đoán hoặc model"}

    df = pd.read_csv(pred_path)
    df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce")
    df = df.dropna().sort_values("Ngày")

    model = load_model(model_path)
    feature_cols = ["open","high","low","close","volume"]

    for c in feature_cols:
        if c not in df.columns:
            df[c] = df["Giá_đóng_cửa_dự_đoán"]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols].values)

    if len(scaled) < 60:
        return {"ok": False, "msg": "Không đủ dữ liệu"}

    seq = scaled[-60:].reshape(1,60,5)
    out = []

    for _ in range(30):
        p = model.predict(seq, verbose=0)
        newc = p[0,3]

        r = seq[:, -1, :].copy()
        r[0,3] = newc
        seq = np.append(seq[:,1:,:], r.reshape(1,1,5), axis=1)

        out.append(newc)

    arr = np.zeros((30,5))
    arr[:,3] = out
    inv = scaler.inverse_transform(arr)[:,3]

    return {"ok": True, "next_month": [float(x) for x in inv]}


@app.get("/predict/next-quarter")
def predict_next_quarter(file: str):

    pred_path = pred_file_path(file)
    model_path = model_file_path(file)

    if not os.path.exists(pred_path) or not os.path.exists(model_path):
        return {"ok": False, "msg": "Thiếu file dự đoán hoặc model"}

    df = pd.read_csv(pred_path)
    df["Ngày"] = pd.to_datetime(df["Ngày"])
    df = df.dropna().sort_values("Ngày")

    model = load_model(model_path)
    feature_cols = ["open","high","low","close","volume"]

    for c in feature_cols:
        if c not in df.columns:
            df[c] = df["Giá_đóng_cửa_dự_đoán"]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols].values)

    if len(scaled) < 60:
        return {"ok": False, "msg": "Không đủ dữ liệu"}

    seq = scaled[-60:].reshape(1,60,5)
    out = []

    for _ in range(90):
        p = model.predict(seq, verbose=0)
        newc = p[0,3]

        r = seq[:,-1,:].copy()
        r[0,3] = newc
        seq = np.append(seq[:,1:,:], r.reshape(1,1,5), axis=1)

        out.append(newc)

    arr = np.zeros((90,5))
    arr[:,3] = out
    inv = scaler.inverse_transform(arr)[:,3]

    return {"ok": True, "next_quarter": [float(x) for x in inv]}
