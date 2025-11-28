/* app/static/js/dashboard_pro.js */
(function () {
  /* ---------------------------------------------
      CONFIG + GLOBAL STATES
  --------------------------------------------- */
  const el = (id) => document.getElementById(id);

  // fallback logo
  const UPLOADED_LOGO = "/mnt/data/30c4cb39-683e-4a5a-98de-0fb77dfe4b17.png";
  const logoEl = el("dpLogo");

  fetch(logoEl.src, { method: "HEAD" }).catch(() => {
    logoEl.src = UPLOADED_LOGO;
  });

  /* =============================================
        EXPORT MODE TRACKER
        default = dự đoán mô hình
        "month"  = dự báo tháng
        "quarter" = dự báo quý
  ============================================= */
  let currentExportMode = "default";
  let cachedMonthForecast = [];
  let cachedQuarterForecast = [];

  /* ---------------------------------------------
      LOAD CSV FILES
  --------------------------------------------- */
  async function loadFiles() {
    try {
      const res = await fetch("/files");
      const js = await res.json();

      const sel = el("fileSelect");
      sel.innerHTML = "";

      if (js && js.files) {
        js.files.forEach((f) => {
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

  /* ---------------------------------------------
      UPLOAD CSV
  --------------------------------------------- */
  el("btnUpload").addEventListener("click", async () => {
    const fi = el("uploadCSV");
    if (!fi.files.length) return alert("Hãy chọn file CSV!");

    const fd = new FormData();
    fd.append("file", fi.files[0]);

    el("lastAction").textContent = "Uploading...";

    try {
      const r = await fetch("/upload", { method: "POST", body: fd });
      const j = await r.json();

      if (!j.ok) throw new Error(j.msg || "Lỗi upload");

      el("lastAction").textContent = "Upload thành công";
      await loadFiles();
      alert("Upload thành công!");
    } catch (err) {
      alert("Upload lỗi: " + err.message);
      el("lastAction").textContent = "Upload thất bại";
    }
  });

  /* ---------------------------------------------
      REFRESH
  --------------------------------------------- */
  el("btnRefresh").addEventListener("click", loadFiles);

  /* ---------------------------------------------
      TRAIN MODEL
  --------------------------------------------- */
  el("btnTrain").addEventListener("click", async () => {
    const file = el("fileSelect").value;
    if (!file) return alert("Chưa chọn file");

    if (!confirm("Bạn chắc muốn huấn luyện mô hình?")) return;

    el("lastAction").textContent = "Đang huấn luyện...";

    const fd = new FormData();
    fd.append("file", file);

    try {
      const r = await fetch("/train", { method: "POST", body: fd });
      const j = await r.json();

      if (!j.ok) throw new Error(j.msg || "Huấn luyện lỗi");

      alert("Huấn luyện hoàn tất!");
      el("lastAction").textContent = "Huấn luyện xong";

      currentExportMode = "default";
      loadChart();
    } catch (err) {
      alert("Lỗi huấn luyện: " + err.message);
      el("lastAction").textContent = "Huấn luyện thất bại";
    }
  });

  /* ---------------------------------------------
      AUTO CHECK PREDICT FILE
  --------------------------------------------- */
  async function autoLoadChart() {
    const file = el("fileSelect").value;
    if (!file) return;

    try {
      const r = await fetch(`/predict/check?file=${encodeURIComponent(file)}`);
      const j = await r.json();

      if (j.exists) loadChart();
      else {
        el("chart").innerHTML =
          '<div class="p-4 muted">Chưa có dữ liệu dự đoán.</div>';
      }
    } catch (e) {
      console.error("autoLoadChart error", e);
    }
  }

  /* ---------------------------------------------
        MAIN CHART (default)
  --------------------------------------------- */
  async function loadChart(rangeDays = 365) {
    currentExportMode = "default";

    const file = el("fileSelect").value;
    if (!file) return alert("Chưa chọn file!");

    el("lastAction").textContent = "Đang tải biểu đồ...";

    try {
      const r = await fetch(
        `/predict/data?file=${encodeURIComponent(file)}&range=${rangeDays}`
      );
      const js = await r.json();

      if (!js.ok) throw new Error(js.msg || "Không lấy được dữ liệu");

      const traces = [
        {
          x: js.dates,
          y: js.real,
          mode: "lines",
          name: "Giá thực tế",
          line: { width: 2, color: "#38bdf8" },
        },
        {
          x: js.dates,
          y: js.pred,
          mode: "lines",
          name: "Giá dự đoán",
          line: { width: 2, dash: "dash", color: "#f97316" },
        },
      ];

      Plotly.newPlot("chart", traces, {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: { title: "Ngày" },
        yaxis: { title: "Giá" },
        hovermode: "x unified",
        margin: { t: 50 },
      });

      el("lastAction").textContent = "Biểu đồ sẵn sàng";
    } catch (err) {
      el("chart").innerHTML =
        `<div class="p-4 muted">Lỗi biểu đồ: ${err.message}</div>`;
      el("lastAction").textContent = "Lỗi!";
    }
  }

  document
    .querySelectorAll(".range-btn")
    .forEach((b) => b.addEventListener("click", () => loadChart(b.dataset.range)));

  /* ---------------------------------------------
        ADVANCED ANALYSIS: MONTH / QUARTER
  --------------------------------------------- */
  el("btnMonthly").addEventListener("click", async () => {
    const f = el("fileSelect").value;
    if (!f) return alert("Chưa chọn file!");

    el("lastAction").textContent = "Phân tích tháng...";

    try {
      const r = await fetch(`/predict/monthly?file=${encodeURIComponent(f)}`);
      const js = await r.json();

      if (!js.ok) throw new Error(js.msg);

      showAdvanced(js.data, "Tháng");
      el("lastAction").textContent = "Hoàn tất phân tích tháng";
    } catch (e) {
      alert(e.message);
    }
  });

  el("btnQuarterly").addEventListener("click", async () => {
    const f = el("fileSelect").value;
    if (!f) return alert("Chưa chọn file!");

    el("lastAction").textContent = "Phân tích quý...";

    try {
      const r = await fetch(`/predict/quarterly?file=${encodeURIComponent(f)}`);
      const js = await r.json();

      if (!js.ok) throw new Error(js.msg);

      showAdvanced(js.data, "Quý");
      el("lastAction").textContent = "Hoàn tất phân tích quý";
    } catch (e) {
      alert(e.message);
    }
  });

  function showAdvanced(data, colName) {
    const holder = el("adv_table_holder");

    let html = `
    <div class="table-responsive">
    <table class="table table-striped table-bordered">
      <thead class="table-dark">
        <tr>
          <th>${colName}</th>
          <th>Giá trung bình</th>
          <th>Giá cao nhất</th>
          <th>Giá thấp nhất</th>
        </tr>
      </thead><tbody>
    `;

    data.forEach((row) => {
      html += `
      <tr>
        <td>${row[colName]}</td>
        <td class="text-end">${Number(row.Trung_bình).toFixed(2)}</td>
        <td class="text-end">${Number(row.Cao_nhất).toFixed(2)}</td>
        <td class="text-end">${Number(row.Thấp_nhất).toFixed(2)}</td>
      </tr>`;
    });

    holder.innerHTML = html + "</tbody></table></div>";
  }

  /* ---------------------------------------------
        SIGNAL SYSTEM
  --------------------------------------------- */
  async function callSignal(path, file, body) {
    const fd = new FormData();
    fd.append("file", file);
    if (body && body.horizon) fd.append("horizon", body.horizon);

    const res = await fetch(path, { method: "POST", body: fd });
    return res.json();
  }

  function renderSignal(data) {
    const pre = el("signalResult");
    const card = el("signalCard");

    if (!data || !data.ok) {
      pre.textContent = JSON.stringify(data, null, 2);
      card.querySelector(".signal-label").textContent = "Không có kết quả";
      return;
    }

    const top = data.overall_signal || "UNKNOWN";
    const score = Number(data.score || 0).toFixed(3);

    pre.textContent = `Overall: ${top}\nScore: ${score}`;

    const label = card.querySelector(".signal-label");
    label.textContent = `${top} • score ${score}`;

    if (top.toLowerCase().includes("buy"))
      label.style.color = "var(--buy)";
    else if (top.toLowerCase().includes("sell"))
      label.style.color = "var(--sell)";
    else label.style.color = "var(--hold)";
  }

  el("btnSignalSimple").addEventListener("click", async () => {
    const f = el("fileSelect").value;
    const j = await callSignal("/signal/simple", f, { horizon: 7 });
    renderSignal(j);
  });

  el("btnSignalAdvanced").addEventListener("click", async () => {
    const f = el("fileSelect").value;
    const j = await callSignal("/signal/advanced", f);
    renderSignal(j);
  });

  el("btnSignalSummary").addEventListener("click", async () => {
    const f = el("fileSelect").value;
    const j = await callSignal("/signal/summary", f);
    renderSignal(j);
  });

  /* ---------------------------------------------
        FORECAST NEXT MONTH / QUARTER
  --------------------------------------------- */
  el("btnNextMonth").addEventListener("click", async () => {
    const f = el("fileSelect").value;
    const r = await fetch(`/predict/next-month?file=${encodeURIComponent(f)}`);
    const j = await r.json();

    if (!j.ok) return alert(j.msg);

    currentExportMode = "month";
    cachedMonthForecast = j.next_month;

    Plotly.newPlot("chart", [
      {
        y: j.next_month,
        mode: "lines",
        name: "Dự báo tháng",
        line: { width: 3 },
      },
    ]);
  });

  el("btnNextQuarter").addEventListener("click", async () => {
    const f = el("fileSelect").value;
    const r = await fetch(
      `/predict/next-quarter?file=${encodeURIComponent(f)}`
    );
    const j = await r.json();

    if (!j.ok) return alert(j.msg);

    currentExportMode = "quarter";
    cachedQuarterForecast = j.next_quarter;

    Plotly.newPlot("chart", [
      {
        y: j.next_quarter,
        mode: "lines",
        name: "Dự báo quý",
        line: { width: 3 },
      },
    ]);
  });

  /* ---------------------------------------------
        EXPORT CSV (SMART MODE)
  --------------------------------------------- */
  el("btnDownloadCSV").addEventListener("click", () => {
    const file = el("fileSelect").value;

    if (currentExportMode === "month") {
      exportArrayCSV(cachedMonthForecast, "du_bao_thang.csv");
      return;
    }

    if (currentExportMode === "quarter") {
      exportArrayCSV(cachedQuarterForecast, "du_bao_quy.csv");
      return;
    }

    const pred = file.replace(".csv", "_du_doan.csv");
    window.open(`/predictions/${encodeURIComponent(pred)}`, "_blank");
  });

  function exportArrayCSV(arr, filename) {
    let csv = "index,value\n";
    arr.forEach((v, i) => (csv += `${i + 1},${v}\n`));

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");

    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
  }

  /* ---------------------------------------------
        THEME MODE
  --------------------------------------------- */
  el("themeToggle").addEventListener("click", () => {
    document.body.classList.toggle("light-mode");
    const ic = el("themeToggle").querySelector("i");

    if (document.body.classList.contains("light-mode"))
      ic.className = "fa fa-sun";
    else ic.className = "fa fa-moon";
  });

})();
