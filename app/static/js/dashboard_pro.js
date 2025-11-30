// app/static/js/dashboard_pro.js
(function () {
  const el = id => document.getElementById(id);

  // fallback logo
  const UPLOADED_LOGO = "/static/img/fallback_logo.png";
  const logoEl = el("dpLogo");
  if (logoEl) {
    fetch(logoEl.src, { method: "HEAD" }).catch(() => { logoEl.src = UPLOADED_LOGO; });
  }

  // modal helpers
  function showModal(htmlMessage, opts = {}) {
    const modal = el("customModal");
    const msg = el("modalMessage");
    const ok = el("modalOk");
    const cancel = el("modalCancel");
    if (!modal) return Promise.resolve(false);

    msg.innerHTML = htmlMessage;
    modal.style.display = "flex";

    return new Promise(resolve => {
      ok.onclick = () => { modal.style.display = "none"; resolve(true); };
      cancel.onclick = () => { modal.style.display = "none"; resolve(false); };

      // if opts.autoClose after ms
      if (opts.autoClose && typeof opts.autoClose === "number") {
        setTimeout(() => { modal.style.display = "none"; resolve(false); }, opts.autoClose);
      }
    });
  }
  document.getElementById("btnDownloadAPI").addEventListener("click", async () => {
    const sym = document.getElementById("stockSymbol").value.trim();
    if (!sym) return alert("Hãy nhập mã!");

    const fd = new FormData();
    fd.append("symbol", sym);

    const r = await fetch("/api/download", { method: "POST", body: fd });
    const j = await r.json();

    if (!j.ok) return alert(j.msg);

    alert(`✔ Tải dữ liệu ${sym} thành công từ API: ${j.source}`);

    await loadFiles(); // load danh sách CSV
  });



  // API download button
  const btnDownloadAPI = el("btnDownloadAPI");
  if (btnDownloadAPI) {
    btnDownloadAPI.addEventListener("click", async () => {
      const sym = el("stockSymbol").value.trim();
      if (!sym) return showModal("Hãy nhập mã chứng khoán (ví dụ: <b>VNM</b>).");

      const fd = new FormData();
      fd.append("symbol", sym);

      try {
        el("lastAction").textContent = "Đang tải dữ liệu từ API...";
        const r = await fetch("/api/download", { method: "POST", body: fd });
        const j = await r.json();
        if (!j.ok) {
          await showModal(`Lỗi: ${j.msg || "Không tải được dữ liệu"}`);
          el("lastAction").textContent = "Lỗi tải dữ liệu";
          return;
        }
        await showModal(`Tải thành công <b>${j.file}</b> từ nguồn <b>${j.source}</b>`, { autoClose: 1800 });
        el("lastAction").textContent = `Tải xong: ${j.file}`;
        await loadFiles();
      } catch (err) {
        console.error(err);
        await showModal("Có lỗi xảy ra khi gọi API.");
        el("lastAction").textContent = "Lỗi";
      }
    });
  }

  // load files into select
  async function loadFiles() {
    try {
      const res = await fetch("/files");
      const js = await res.json();
      const sel = el("fileSelect");
      if (!sel) return;
      sel.innerHTML = "";
      if (js && js.files) {
        js.files.forEach(f => {
          const opt = document.createElement("option");
          opt.value = f;
          opt.textContent = f;
          sel.appendChild(opt);
        });
      }
      autoLoadChart();
    } catch (e) {
      console.error("loadFiles error", e);
    }
  }
  loadFiles();

  // upload
  el("btnUpload")?.addEventListener("click", async () => {
    const fi = el("uploadCSV");
    if (!fi || !fi.files.length) return showModal("Hãy chọn file CSV để upload.");
    const fd = new FormData();
    fd.append("file", fi.files[0]);
    el("lastAction").textContent = "Uploading...";
    try {
      const r = await fetch("/upload", { method: "POST", body: fd });
      const j = await r.json();
      if (!j.ok) throw new Error(j.msg || "Lỗi upload");
      el("lastAction").textContent = "Upload thành công";
      await loadFiles();
      await showModal("Upload thành công!", { autoClose: 1200 });
    } catch (err) {
      await showModal("Upload lỗi: " + (err.message || err));
      el("lastAction").textContent = "Upload thất bại";
    }
  });

  el("btnRefresh")?.addEventListener("click", loadFiles);

  // train
  el("btnTrain")?.addEventListener("click", async () => {
    const file = el("fileSelect")?.value;
    if (!file) return showModal("Chưa chọn file!");
    el("lastAction").textContent = "Đang kiểm tra mô hình...";
    const fd = new FormData();
    fd.append("file", file);
    fd.append("force", 0);
    try {
      const r = await fetch("/train", { method: "POST", body: fd });
      const j = await r.json();
      if (j.need_confirm) {
        const userChoice = await showModal(
          `File "<b>${file}</b>" đã được train trước đó.<br>Bạn có muốn <b style="color:#38bdf8">TRAIN LẠI</b> không?`, {}
        );
        if (!userChoice) {
          await showModal("Sử dụng model cũ.", { autoClose: 1000 });
          el("lastAction").textContent = "Đang dùng model cũ";
          loadChart();
          return;
        }
        fd.set("force", 1);
        el("lastAction").textContent = "Đang train lại mô hình...";
        const r2 = await fetch("/train", { method: "POST", body: fd });
        const j2 = await r2.json();
        await showModal(j2.msg || "Train xong", { autoClose: 1200 });
        el("lastAction").textContent = "Train lại hoàn tất";
        loadChart();
        return;
      }
      await showModal(j.msg || "Train thành công", { autoClose: 1200 });
      el("lastAction").textContent = "Huấn luyện xong";
      loadChart();
    } catch (err) {
      console.error(err);
      await showModal("Lỗi khi huấn luyện: " + (err.message || err));
      el("lastAction").textContent = "Huấn luyện thất bại";
    }
  });




  async function autoLoadChart() {
    const file = el("fileSelect")?.value;
    if (!file) return;
    try {
      const r = await fetch(`/predict/check?file=${encodeURIComponent(file)}`);
      const j = await r.json();
      if (j.exists) loadChart();
      else el("chart").innerHTML = '<div class="p-4 muted">Chưa có dữ liệu dự đoán.</div>';
    } catch (e) { console.error("autoLoadChart error", e); }
  }

  async function loadChart(rangeDays = 365) {
    const file = el("fileSelect")?.value;
    if (!file) return;

    const start = el("startDate")?.value;
    const end = el("endDate")?.value;

    el("lastAction").textContent = "Đang tải biểu đồ...";

    try {
      // FIX QUAN TRỌNG: đổi range → range_days
      let url = `/predict/data?file=${encodeURIComponent(file)}&range_days=${rangeDays}`;

      if (start) url += `&start_date=${start}`;
      if (end) url += `&end_date=${end}`;

      console.log("API:", url);

      const r = await fetch(url);
      const js = await r.json();
      if (!js.ok) throw new Error(js.msg || "Không lấy được dữ liệu");

      const traces = [
        { x: js.dates, y: js.real, mode: "lines", name: "Giá thực tế", line: { width: 2, color: "#38bdf8" } },
        { x: js.dates, y: js.pred, mode: "lines", name: "Giá dự đoán", line: { width: 2, dash: "dash", color: "#f97316" } }
      ];

      Plotly.newPlot("chart", traces, {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: { title: "Ngày" },
        yaxis: { title: "Giá" },
        hovermode: "x unified",
        margin: { t: 50 }
      });

      el("lastAction").textContent = "Biểu đồ sẵn sàng";
    } catch (err) {
      el("chart").innerHTML = `<div class="p-4 muted">Lỗi biểu đồ: ${err.message}</div>`;
      el("lastAction").textContent = "Lỗi!";
    }
  }

  el("btnApplyRange")?.addEventListener("click", () => {
    loadChart();
  });



  document.querySelectorAll(".range-btn").forEach(b => b.addEventListener("click", () => loadChart(b.dataset.range)));

  // signal helpers (same as before)
  async function callSignal(path, file, body = {}) {
    const fd = new FormData();
    fd.append("file", file);

    const start = document.getElementById("startDate").value;
    const end = document.getElementById("endDate").value;

    if (start) fd.append("start_date", start);
    if (end) fd.append("end_date", end);

    if (body.horizon) fd.append("horizon", body.horizon);

    return (await fetch(path, { method: "POST", body: fd })).json();
  }

  function renderSignal(data) {
    const pre = el("signalResult");
    const card = el("signalCard");

    if (!data || !data.ok) {
        pre.textContent = JSON.stringify(data, null, 2);
        card.querySelector(".signal-label").textContent = "Không có kết quả";
        return;
    }

    // Ưu tiên thứ tự: Summary → Advanced → Simple
    const signal = data.overall_signal || data.signal || "UNKNOWN";
    const score = Number(data.score || 0).toFixed(3);

    // Khối nội dung mô tả ở dưới
    pre.textContent = `Overall: ${signal}\nScore: ${score}`;

    // Dòng chữ lớn phía trên
    const label = card.querySelector(".signal-label");
    label.textContent = `${signal} • score ${score}`;

    // Set màu theo tín hiệu
    const s = signal.toLowerCase();
    if (s.includes("buy")) label.style.color = "var(--buy)";
    else if (s.includes("sell")) label.style.color = "var(--sell)";
    else label.style.color = "var(--hold)";
}


  el("btnSignalSimple")?.addEventListener("click", async () => {
    const f = el("fileSelect").value;
    const j = await callSignal("/signal/simple", f, { horizon: 7 });
    renderSignal(j);
  });

  el("btnSignalAdvanced")?.addEventListener("click", async () => {
    const f = el("fileSelect").value;
    if (!f) return showModal("Chưa chọn file!");
    const j = await callSignal("/signal/advanced", f);
    renderSignal(j);
    renderAdvancedDaily(j);
  });

  el("btnSignalSummary")?.addEventListener("click", async () => {
    const f = el("fileSelect").value;
    const j = await callSignal("/signal/summary", f);
    renderSignal(j);
    renderAdvancedDaily(j);
  });

  el("btnSignalRecommend")?.addEventListener("click", async () => {
    const f = el("fileSelect").value;
    if (!f) return showModal("Chưa chọn file!");
    const j = await callSignal("/signal/recommend", f);
    renderSignal(j);
  });

  el("btnNextMonth")?.addEventListener("click", async () => {
    const f = el("fileSelect").value;
    const r = await fetch(`/predict/next-month?file=${encodeURIComponent(f)}`);
    const j = await r.json();
    if (!j.ok) return showModal(j.msg || "Không có dữ liệu");
    Plotly.newPlot("chart", [{ y: j.next_month, mode: "lines", name: "Dự báo tháng", line: { width: 3 } }]);
  });

  el("btnNextQuarter")?.addEventListener("click", async () => {
    const f = el("fileSelect").value;
    const r = await fetch(`/predict/next-quarter?file=${encodeURIComponent(f)}`);
    const j = await r.json();
    if (!j.ok) return showModal(j.msg || "Không có dữ liệu");
    Plotly.newPlot("chart", [{ y: j.next_quarter, mode: "lines", name: "Dự báo quý", line: { width: 3 } }]);
  });

  el("btnDownloadCSV")?.addEventListener("click", () => {
    const file = el("fileSelect").value;
    const pred = file.replace(".csv", "_du_doan.csv");
    window.open(`/predictions/${encodeURIComponent(pred)}`, "_blank");
  });

  el("themeToggle")?.addEventListener("click", () => {
    document.body.classList.toggle("light-mode");
    const ic = el("themeToggle").querySelector("i");
    if (document.body.classList.contains("light-mode")) ic.className = "fa fa-sun";
    else ic.className = "fa fa-moon";
  });

// --- thêm cuối file hoặc thay handler hiện tại ---

// render table daily (kèm style nhẹ)
function renderAdvancedDaily(data) {
    const box = document.getElementById("advDailyBox");
    if (!box) return;

    if (!data.daily || data.daily.length === 0) {
        box.innerHTML = `
            <div style="padding:12px; background:#111; color:#ccc; text-align:center; border-radius:8px;">
                Không có dữ liệu hàng ngày để hiển thị.
            </div>`;
        return;
    }

    let html = `
        <table class="adv-table">
            <thead>
                <tr>
                    <th>Ngày</th>
                    <th>Giá</th>
                    <th>MA7</th>
                    <th>MA14</th>
                    <th>MA50</th>
                    <th>RSI</th>
                    <th>MACD</th>
                    <th>ROC</th>
                </tr>
            </thead>
            <tbody>
    `;

    data.daily.forEach(row => {
        const rsiClass = row.rsi > 70 ? "red" : row.rsi < 30 ? "green" : "yellow";
        const macdClass = row.macd > 0 ? "green" : "red";
        const rocClass = row.roc > 0 ? "green" : "red";

        html += `
            <tr>
                <td>${row.date}</td>
                <td class="price">${row.close.toFixed(2)}</td>
                <td>${row.ma7 ? row.ma7.toFixed(2) : "-"}</td>
                <td>${row.ma14 ? row.ma14.toFixed(2) : "-"}</td>
                <td>${row.ma50 ? row.ma50.toFixed(2) : "-"}</td>
                <td class="${rsiClass}">${row.rsi ? row.rsi.toFixed(2) : "-"}</td>
                <td class="${macdClass}">${row.macd ? row.macd.toFixed(2) : "-"}</td>
                <td class="${rocClass}">${row.roc ? row.roc.toFixed(2) : "-"}</td>
            </tr>
        `;
    });

    html += "</tbody></table>";
    box.innerHTML = html;
}


// btnSignalSummary: gọi API và show cả summary + bảng hàng ngày
el("btnSignalSummary")?.addEventListener("click", async () => {
  const f = el("fileSelect").value;
  if (!f) return showModal("Chưa chọn file!");
  try {
    const j = await callSignal("/signal/summary", f);
    // render label + score
    renderSignal(j);

    // render daily bảng nâng cao nếu có
    renderAdvancedDaily(j);
  } catch (err) {
    console.error(err);
    showModal("Không thể lấy summary!");
  }
});

})();
