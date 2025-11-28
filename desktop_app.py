import subprocess
import time
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl,Qt
import os

# Thêm thư mục app vào sys.path để import được main
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # = DATN]
APP_DIR = os.path.join(BASE_DIR, "app")
sys.path.insert(0, BASE_DIR)

# Chạy FastAPI bằng Uvicorn
# Lưu ý: main:app trỏ tới file main.py trong folder app
FASTAPI_CMD = [
    sys.executable, "-m", "uvicorn", "app.main:app",
    "--host", "127.0.0.1",
    "--port", "8000",
    "--reload"
]
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1400, 900)
        self.setWindowTitle("Dự Đoán Giá Chứng Khoán")
        view = QWebEngineView()
        view.setUrl(QUrl("http://127.0.0.1:8000/dashboard"))
        self.setCentralWidget(view)

if __name__ == "__main__":
    # Khởi chạy FastAPI
    fastapi_process = subprocess.Popen(FASTAPI_CMD, cwd=BASE_DIR)


    # Đợi server chạy
    time.sleep(3)

    # Mở app desktop
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

    # Tắt server khi đóng app
    fastapi_process.terminate()
