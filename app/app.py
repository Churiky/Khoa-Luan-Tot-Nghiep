from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from train_model import train_lstm
import threading

app = Flask(__name__)
app.secret_key = "secret_key_demo"

# ====== Đường dẫn ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "DATA")
PRED_DIR = os.path.join(PROJECT_ROOT, "DATA_PREDICT")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# ====== Trạng thái train ======
train_state = {"running": False, "progress": 0, "message": "", "rmse": None, "mape": None, "mae": None}
train_lock = threading.Lock()

def set_state(**kwargs):
    with train_lock:
        train_state.update(kwargs)

def get_state():
    with train_lock:
        return dict(train_state)

# ====== Liệt kê file ======
def list_data_files():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    return sorted(files, key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)), reverse=True)

def get_pred_path_for_local(data_filename):
    base = os.path.splitext(os.path.basename(data_filename))[0]
    return os.path.join(PRED_DIR, f"{base}_du_doan.csv")

# ===================== UPLOAD CSV ===================== #
@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    file = request.files.get("file")
    if not file:
        flash("❌ Không có file được chọn.", "danger")
        return redirect(url_for("dashboard"))

    filename = file.filename
    save_path = os.path.join(DATA_DIR, filename)

    try:
        df = pd.read_csv(file)
        df.columns = [c.strip().lower() for c in df.columns]
        mapping = {
            "time": "date", "ngay": "date",
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume",
            "kl": "volume", "gia_dong_cua": "close"
        }
        df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})
        for col in ["date", "open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        flash(f"✅ File '{filename}' đã được chuẩn hóa và lưu vào thư mục DATA!", "success")
    except Exception as e:
        flash(f"❌ Lỗi khi xử lý file: {e}", "danger")

    return redirect(url_for("dashboard"))

# ===================== TRAIN ===================== #
def background_train(data_path, start_date=None, end_date=None, loss_function="mse"):
    try:
        set_state(running=True, progress=0, message="Đang huấn luyện mô hình...")
        result = train_lstm(
            data_path,
            start_date=start_date,
            end_date=end_date,
            loss_function=loss_function,
            progress_callback=lambda p, m=None: set_state(progress=p, message=m or "")
        )
        if result:
            _, mae, rmse = result 
            set_state(running=False, progress=100, message="Huấn luyện hoàn tất!", 
                      rmse=rmse, mae=mae, mape=None)
        else:
            set_state(running=False, message="Huấn luyện thất bại!", rmse=None, mae=None, mape=None)
    except Exception as e:
        set_state(running=False, message=f"❌ Lỗi huấn luyện: {e}")

@app.route("/start_train", methods=["POST"])
def start_train():
    file = request.form.get("selected_file")
    start_date = request.form.get("start_date") or None
    end_date = request.form.get("end_date") or None
    loss_function = request.form.get("loss_function", "mse")
    if not file:
        return jsonify({"ok": False, "msg": "Chưa chọn file!"}), 400
    data_path = os.path.join(DATA_DIR, file)
    pred_path = get_pred_path_for_local(file)

    # Kiểm tra nếu file dự đoán đã có và nằm trong khoảng time user nhập
    if os.path.exists(pred_path) and start_date and end_date:
        try:
            pred_df = pd.read_csv(pred_path)
            pred_df["Ngày"] = pd.to_datetime(pred_df["Ngày"], errors="coerce")
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            if pred_df["Ngày"].min() <= start_dt and pred_df["Ngày"].max() >= end_dt:
                # Trả redirect về dashboard với start_date / end_date
                return jsonify({
                    "ok": True,
                    "msg": "✅ Dữ liệu dự đoán đã có, không cần train lại.",
                    "redirect": url_for("dashboard", selected_file=os.path.basename(data_path),
                                        start_date=start_date, end_date=end_date)
                })
        except:
            pass

    if not os.path.exists(data_path):
        return jsonify({"ok": False, "msg": "File không tồn tại!"}), 404

    thread = threading.Thread(target=background_train, args=(data_path, start_date, end_date, loss_function))
    thread.start()
    return jsonify({"ok": True})

@app.route("/train_status")
def train_status():
    return jsonify(get_state())

# ===================== DASHBOARD ===================== #
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "admin" and password == "123":
            session["username"] = username
            return redirect(url_for("dashboard"))
        else:
            flash("Sai tài khoản hoặc mật khẩu!", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard", methods=["GET"])
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))

    files = list_data_files()
    if not files:
        # ... (code xử lý không có file của bạn giữ nguyên) ...
        return render_template(
            "dashboard.html",
            error="❌ Không có dữ liệu trong thư mục DATA.",
            files=[],
            selected_file=None,
            plot_html="",
            table_data="",
            pred_preview_html="",
            train_state=get_state(),
            start_date=None,
            end_date=None
        )

    # --- Lấy dữ liệu từ query params / form ---
    selected = request.args.get("selected_file", files[0])
    
    # 1. Lấy ngày tháng GỐC từ người dùng
    user_start_date = request.args.get("start_date", None)
    user_end_date = request.args.get("end_date", None)

    data_path = os.path.join(DATA_DIR, selected)
    pred_path = get_pred_path_for_local(selected)

    try:
        # ... (code đọc df của bạn giữ nguyên) ...
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.lower()
        df = df.rename(columns={"time": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    except Exception as e:
        # ... (code xử lý lỗi đọc file của bạn giữ nguyên) ...
        return render_template(
            "dashboard.html",
            error=f"Lỗi đọc dữ liệu: {e}",
            files=files,
            selected_file=selected,
            # ... (phần còn lại)
        )

    pred_df = None
    if os.path.exists(pred_path):
        try:
            pred_df = pd.read_csv(pred_path)
        except Exception as e:
            print("Lỗi đọc file dự đoán:", e)

    # ===== Vẽ biểu đồ =====
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["close"],
        mode="lines",
        name="Giá thật",
        line=dict(color="green")
    ))

    # 2. Khởi tạo biến ngày tháng *cuối cùng* để vẽ
    # Mặc định là ngày người dùng chọn
    final_start_date = user_start_date
    final_end_date = user_end_date

    if pred_df is not None and "Ngày" in pred_df.columns:
        pred_df["Ngày"] = pd.to_datetime(pred_df["Ngày"], errors="coerce")
        last_real_date = df["date"].max()
        future_df = pred_df[pred_df["Ngày"] > last_real_date]
        past_df = pred_df[pred_df["Ngày"] <= last_real_date]

        if len(past_df) > 0:
            fig.add_trace(go.Scatter(
                x=past_df["Ngày"],
                y=past_df["Giá_đóng_cửa_dự_đoán"],
                mode="lines+markers",
                name="Dự đoán (trong mẫu)",
                line=dict(color="red", dash="dash")
            ))
        if len(future_df) > 0:
            fig.add_trace(go.Scatter(
                x=future_df["Ngày"],
                y=future_df["Giá_đóng_cửa_dự_đoán"],
                mode="lines+markers",
                name="Dự đoán tương lai",
                line=dict(color="orange", width=3)
            ))

            # 3. LOGIC SỬA: Chỉ auto-zoom NẾU người dùng không chọn ngày
            if not user_start_date and not user_end_date:
                zoom_start = last_real_date - pd.Timedelta(days=20)
                zoom_end = future_df["Ngày"].max() + pd.Timedelta(days=5)
                
                final_start_date = zoom_start.strftime("%Y-%m-%d")
                final_end_date = zoom_end.strftime("%Y-%m-%d")

    # 4. Tính toán X_RANGE dựa trên ngày tháng *cuối cùng*
    if final_start_date and final_end_date:
        # Đây là logic để tạo padding 20 ngày (tạo ra Ảnh 2)
        x_range_start = pd.to_datetime(final_start_date) - pd.Timedelta(days=20)
        x_range_end = pd.to_datetime(final_end_date)
    else:
        # Nếu không có ngày nào (lần đầu load), hiển thị toàn bộ (Ảnh 1)
        x_range_start = df["date"].min()
        x_range_end = df["date"].max()

    fig.update_layout(
        title=f"📈 Dự đoán giá cổ phiếu – {selected}",
        xaxis_title="Ngày",
        yaxis_title="Giá (VND)",
        xaxis=dict(range=[x_range_start, x_range_end]), # Dùng range đã tính
        template="plotly_white",
        height=600
    )

    plot_html = pio.to_html(fig, full_html=False)
    table_data = df.tail(20).to_html(classes="table table-striped", index=False)
    pred_preview_html = pred_df.tail(20).to_html(classes="table table-striped", index=False) if pred_df is not None else ""

    return render_template(
        "dashboard.html",
        username=session.get("username"),
        files=files,
        selected_file=selected,
        plot_html=plot_html,
        table_data=table_data,
        pred_preview_html=pred_preview_html,
        train_state=get_state(),
        # 5. Trả về ngày tháng đã dùng để điền vào form
        start_date=final_start_date,
        end_date=final_end_date
    )


    

if __name__ == "__main__":
    app.run(debug=True)
