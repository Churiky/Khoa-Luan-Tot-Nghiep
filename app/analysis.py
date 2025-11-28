# app/analysis.py
import os
import pandas as pd

def get_predict_path(base_dir, file):
    return os.path.join(base_dir, "DATA_PREDICT", file.replace(".csv", "_du_doan.csv"))

def analyze_monthly(pred_path):
    if not os.path.exists(pred_path):
        return None, "Chưa có file dự đoán"
    try:
        df = pd.read_csv(pred_path)
        df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce")
        df = df.dropna(subset=["Ngày", "Giá_đóng_cửa_dự_đoán"])
        df["Tháng"] = df["Ngày"].dt.to_period("M").astype(str)
        grouped = df.groupby("Tháng")["Giá_đóng_cửa_dự_đoán"].agg(
            Trung_bình="mean", Cao_nhất="max", Thấp_nhất="min"
        ).reset_index()
        return grouped.to_dict(orient="records"), None
    except Exception as e:
        return None, f"Lỗi phân tích theo tháng: {str(e)}"

def analyze_quarterly(pred_path):
    if not os.path.exists(pred_path):
        return None, "Chưa có file dự đoán"
    try:
        df = pd.read_csv(pred_path)
        df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce")
        df = df.dropna(subset=["Ngày", "Giá_đóng_cửa_dự_đoán"])
        df["Quý"] = df["Ngày"].dt.to_period("Q").astype(str)
        grouped = df.groupby("Quý")["Giá_đóng_cửa_dự_đoán"].agg(
            Trung_bình="mean", Cao_nhất="max", Thấp_nhất="min"
        ).reset_index()
        return grouped.to_dict(orient="records"), None
    except Exception as e:
        return None, f"Lỗi phân tích theo quý: {str(e)}"
