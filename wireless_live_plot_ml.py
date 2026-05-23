import sys
import os
import csv
import socket
import threading
import datetime
import numpy as np
from collections import deque

import joblib
from sklearn.ensemble import RandomForestClassifier

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui


#c:\Users\838bl\Documents\VSCode\CAPSTONE\.venv\Scripts\python.exe "c:/Users/838bl/Documents/VSCode/CAPSTONE WIRELESS/wireless_live_plot_ml.py"
# ---- ML MODEL ----
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    _model = joblib.load(os.path.join(MODEL_DIR, "surface_model.pkl"))
    _le    = joblib.load(os.path.join(MODEL_DIR, "surface_labels.pkl"))
    ML_READY = True
    print(f"Model loaded. Classes: {list(_le.classes_)}")
except Exception as e:
    ML_READY = False
    print(f"No model found ({e}) — classification disabled")

# ---- CONFIG ----
UDP_IP      = "0.0.0.0"
UDP_PORT    = 4210
NUM_SENSORS = 8
HISTORY     = 80
SAVE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(SAVE_DIR, exist_ok=True)

LABELS = [
    "F1_405", "F2_425", "FZ_450", "F3_475", "F4_515",
    "F5_550", "FY_555", "FXL_600", "F6_640",
    "F7_690", "F8_745", "NIR_855"
]
LABELS_SHORT = [
    "F1\n405", "F2\n425", "FZ\n450", "F3\n475", "F4\n515",
    "F5\n550", "FY\n555", "FXL\n600", "F6\n640",
    "F7\n690", "F8\n745", "NIR\n855"
]
N = len(LABELS)
WAVELENGTHS = [405, 425, 450, 475, 515, 550, 555, 600, 640, 690, 745, 855]

COLORS = [
    (106, 13, 173), (127, 0, 255), (0, 71, 255), (0, 183, 235), (57, 255, 20),
    (173, 255, 47), (191, 255, 0), (255, 192, 0), (255, 59, 0),
    (255, 0, 0), (139, 11, 11), (30, 30, 30)
]

# ---- SHARED STATE ----
latest   = [[0]*N for _ in range(NUM_SENSORS)]
smooth   = [[0.0]*N for _ in range(NUM_SENSORS)]
hist     = [[deque([0]*HISTORY, maxlen=HISTORY) for _ in range(N)]
            for _ in range(NUM_SENSORS)]
hist_avg = [deque([0]*HISTORY, maxlen=HISTORY) for _ in range(N)]
lock     = threading.Lock()
LERP     = 0.2

# ---- DATA COLLECTION STATE ----
recording     = False
record_lock   = threading.Lock()
sample_count  = 0
csv_writer    = None
csv_file      = None

# ---- SPECTRAL → RGB ----
def wavelength_to_rgb(wl):
    if 380 <= wl < 440:
        r, g, b = -(wl - 440) / 60, 0, 1
    elif 440 <= wl < 490:
        r, g, b = 0, (wl - 440) / 50, 1
    elif 490 <= wl < 510:
        r, g, b = 0, 1, -(wl - 510) / 20
    elif 510 <= wl < 580:
        r, g, b = (wl - 510) / 70, 1, 0
    elif 580 <= wl < 645:
        r, g, b = 1, -(wl - 645) / 65, 0
    elif 645 <= wl <= 780:
        r, g, b = 1, 0, 0
    else:
        r, g, b = 0.3, 0.3, 0.3
    return r, g, b

def spectral_to_rgb(values):
    total = sum(values) or 1
    r, g, b = 0, 0, 0
    for wl, val in zip(WAVELENGTHS, values):
        weight = val / total
        wr, wg, wb = wavelength_to_rgb(wl)
        r += wr * weight
        g += wg * weight
        b += wb * weight
    mx = max(r, g, b, 0.001)
    r, g, b = r/mx, g/mx, b/mx
    gamma = 0.8
    return (max(0, min(255, int(r**gamma * 255))),
            max(0, min(255, int(g**gamma * 255))),
            max(0, min(255, int(b**gamma * 255))))

# ---- UDP READER THREAD ----
def serial_reader():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening for UDP on port {UDP_PORT}")

    while True:
        try:
            data, _ = sock.recvfrom(256)
            line = data.decode("utf-8", errors="replace").strip()
            if not line or "," not in line: continue
            if len(line) < 3 or line[0] != "S" or ":" not in line: continue
            try:
                idx = int(line[1])
            except ValueError:
                continue
            if idx < 0 or idx >= NUM_SENSORS: continue
            parts = line[3:].split(",")
            if len(parts) < N: continue
            values = list(map(int, parts[:N]))
            with lock:
                latest[idx][:] = values
                for j, v in enumerate(values):
                    hist[idx][j].append(v)
        except Exception:
            pass

thread = threading.Thread(target=serial_reader, daemon=True)
thread.start()

# ---- PyQtGraph SETUP ----
pg.setConfigOptions(antialias=False, background='#1e1e1e', foreground='#dddddd')
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
app = QtWidgets.QApplication(sys.argv)
app.setStyleSheet("""
    QWidget { background-color: #1e1e1e; color: #dddddd; }
    QLabel  { color: #dddddd; }
    QFrame#sensorBox {
        border: 1px solid #444444;
        border-radius: 6px;
        background-color: #252525;
    }
    QFrame#recordBar {
        border: 1px solid #555555;
        border-radius: 8px;
        background-color: #2a2a2a;
    }
    QLineEdit {
        background-color: #333333;
        color: #ffffff;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 13px;
    }
    QPushButton {
        border-radius: 5px;
        padding: 6px 18px;
        font-size: 13px;
        font-weight: bold;
    }
    QPushButton#startBtn {
        background-color: #2e7d32;
        color: white;
        border: none;
    }
    QPushButton#startBtn:hover { background-color: #388e3c; }
    QPushButton#stopBtn {
        background-color: #c62828;
        color: white;
        border: none;
    }
    QPushButton#stopBtn:hover { background-color: #d32f2f; }
    QPushButton#stopBtn:disabled {
        background-color: #444444;
        color: #777777;
    }
    QPushButton#startBtn:disabled {
        background-color: #444444;
        color: #777777;
    }
""")

win = QtWidgets.QMainWindow()
win.setWindowTitle("Live Spectral Dashboard  —  8× AS7343  [WiFi]")
win.resize(1700, 1020)

# Main layout: sensor grid on top, record bar on bottom
main_widget = QtWidgets.QWidget()
win.setCentralWidget(main_widget)
main_vl = QtWidgets.QVBoxLayout(main_widget)
main_vl.setSpacing(6)
main_vl.setContentsMargins(6, 6, 6, 6)

# ── Sensor grid ────────────────────────────────────────────────────────────────
grid_widget = QtWidgets.QWidget()
grid = QtWidgets.QGridLayout(grid_widget)
grid.setSpacing(6)
grid.setContentsMargins(0, 0, 0, 0)
grid.setRowStretch(0, 1)
grid.setRowStretch(1, 1)
grid.setRowStretch(2, 1)
grid.setColumnStretch(0, 1)
grid.setColumnStretch(1, 1)
grid.setColumnStretch(2, 1)
grid.setColumnStretch(3, 1)
main_vl.addWidget(grid_widget, stretch=1)

x_line = np.arange(HISTORY)

bar_items     = []
line_items    = []
bar_plots     = []
line_plots    = []
swatch_labels = []
rgb_labels    = []
value_labels  = []

def make_sensor_panel(sensor_idx, title):
    frame = QtWidgets.QFrame()
    frame.setObjectName("sensorBox")
    vl = QtWidgets.QVBoxLayout(frame)
    vl.setSpacing(3)
    vl.setContentsMargins(6, 6, 6, 6)

    bp = pg.PlotWidget(title=title)
    bp.setMenuEnabled(False)
    bp.showGrid(x=False, y=True, alpha=0.3)
    bp.setYRange(0, 1000)
    bp.getAxis('bottom').setTicks([list(enumerate(LABELS_SHORT))])
    bp.getAxis('bottom').setStyle(tickFont=pg.QtGui.QFont('Arial', 5))
    vl.addWidget(bp, stretch=3)

    lp = pg.PlotWidget()
    lp.setMenuEnabled(False)
    lp.showGrid(x=True, y=True, alpha=0.3)
    lp.setXRange(0, HISTORY-1)
    lp.setYRange(0, 1000)
    vl.addWidget(lp, stretch=2)

    info = QtWidgets.QWidget()
    info_vl = QtWidgets.QVBoxLayout(info)
    info_vl.setContentsMargins(0, 0, 0, 0)
    info_vl.setSpacing(2)

    top_row = QtWidgets.QWidget()
    top_hl = QtWidgets.QHBoxLayout(top_row)
    top_hl.setContentsMargins(0, 0, 0, 0)
    top_hl.setSpacing(6)

    swatch = QtWidgets.QLabel()
    swatch.setFixedSize(36, 36)
    swatch.setStyleSheet("background-color: rgb(100,100,100); border: 1px solid #555; border-radius: 4px;")
    top_hl.addWidget(swatch)

    rgb_lbl = QtWidgets.QLabel("rgb(—,—,—)")
    rgb_lbl.setStyleSheet("font-size: 10px; font-family: Consolas, monospace; color: #aaaaaa;")
    top_hl.addWidget(rgb_lbl)
    top_hl.addStretch()
    info_vl.addWidget(top_row)

    vals_lbl = QtWidgets.QLabel("—")
    vals_lbl.setStyleSheet("font-size: 12px; font-family: Consolas, monospace; color: #eeeeee; font-weight: bold;")
    vals_lbl.setWordWrap(False)
    info_vl.addWidget(vals_lbl)
    vl.addWidget(info)

    bis = []
    for j in range(N):
        bi = pg.BarGraphItem(x=[j], height=[0], width=0.7, brush=pg.mkBrush(*COLORS[j]))
        bp.addItem(bi)
        bis.append(bi)

    lis = []
    for j in range(N):
        li = lp.plot(x_line, np.zeros(HISTORY), pen=pg.mkPen(color=COLORS[j], width=1))
        lis.append(li)

    return frame, bp, lp, bis, lis, swatch, rgb_lbl, vals_lbl

for i in range(NUM_SENSORS):
    row = 0 if i < 4 else 1
    col = i if i < 4 else i - 4
    frame, bp, lp, bis, lis, swatch, rgb_lbl, vals_lbl = make_sensor_panel(i, f"Sensor {i}  (ch {i})")
    grid.addWidget(frame, row, col)
    bar_plots.append(bp);     line_plots.append(lp)
    bar_items.append(bis);    line_items.append(lis)
    swatch_labels.append(swatch); rgb_labels.append(rgb_lbl); value_labels.append(vals_lbl)

# ── Averaged panel ─────────────────────────────────────────────────────────────
avg_frame = QtWidgets.QFrame()
avg_frame.setObjectName("sensorBox")
avg_frame.setMaximumHeight(400)
avg_hl = QtWidgets.QHBoxLayout(avg_frame)
avg_hl.setSpacing(6)
avg_hl.setContentsMargins(6, 6, 6, 6)

avg_charts = QtWidgets.QWidget()
avg_charts_vl = QtWidgets.QVBoxLayout(avg_charts)
avg_charts_vl.setSpacing(3)
avg_charts_vl.setContentsMargins(0, 0, 0, 0)

bp_avg = pg.PlotWidget(title="Average  —  All 8 Sensors")
bp_avg.setMenuEnabled(False)
bp_avg.showGrid(x=False, y=True, alpha=0.3)
bp_avg.setYRange(0, 1000)
bp_avg.getAxis('bottom').setTicks([list(enumerate(LABELS_SHORT))])
bp_avg.getAxis('bottom').setStyle(tickFont=pg.QtGui.QFont('Arial', 6))
avg_charts_vl.addWidget(bp_avg, stretch=3)

lp_avg = pg.PlotWidget()
lp_avg.setMenuEnabled(False)
lp_avg.showGrid(x=True, y=True, alpha=0.3)
lp_avg.setXRange(0, HISTORY-1)
lp_avg.setYRange(0, 1000)
avg_charts_vl.addWidget(lp_avg, stretch=2)
avg_hl.addWidget(avg_charts, stretch=4)

avg_right = QtWidgets.QWidget()
avg_right_vl = QtWidgets.QVBoxLayout(avg_right)
avg_right_vl.setContentsMargins(4, 4, 4, 4)
avg_right_vl.setSpacing(6)
avg_right_vl.setAlignment(QtCore.Qt.AlignTop)

avg_title = QtWidgets.QLabel("AVG Color\n(All 8)")
avg_title.setAlignment(QtCore.Qt.AlignCenter)
avg_title.setStyleSheet("font-weight: bold; font-size: 12px;")
avg_right_vl.addWidget(avg_title)

avg_swatch = QtWidgets.QLabel()
avg_swatch.setFixedSize(80, 80)
avg_swatch.setStyleSheet("background-color: rgb(100,100,100); border: 2px solid #555; border-radius: 8px;")
avg_right_vl.addWidget(avg_swatch, alignment=QtCore.Qt.AlignCenter)

avg_rgb_label = QtWidgets.QLabel("rgb(—,—,—)")
avg_rgb_label.setAlignment(QtCore.Qt.AlignCenter)
avg_rgb_label.setStyleSheet("font-size: 11px; font-family: Consolas, monospace; color: #aaaaaa;")
avg_right_vl.addWidget(avg_rgb_label)

avg_vals_label = QtWidgets.QLabel("—")
avg_vals_label.setStyleSheet("font-size: 12px; font-family: Consolas, monospace; color: #eeeeee;")
avg_vals_label.setWordWrap(True)
avg_right_vl.addWidget(avg_vals_label)
avg_hl.addWidget(avg_right, stretch=1)

grid.addWidget(avg_frame, 2, 1, 1, 2)

bis_avg = []
for j in range(N):
    bi = pg.BarGraphItem(x=[j], height=[0], width=0.7, brush=pg.mkBrush(*COLORS[j]))
    bp_avg.addItem(bi)
    bis_avg.append(bi)

lis_avg = []
for j in range(N):
    li = lp_avg.plot(x_line, np.zeros(HISTORY), pen=pg.mkPen(color=COLORS[j], width=1.2))
    lis_avg.append(li)

# ── Data Collection Bar ────────────────────────────────────────────────────────
record_frame = QtWidgets.QFrame()
record_frame.setObjectName("recordBar")
record_frame.setMaximumHeight(70)
rec_hl = QtWidgets.QHBoxLayout(record_frame)
rec_hl.setContentsMargins(12, 8, 12, 8)
rec_hl.setSpacing(12)

# Label
rec_label = QtWidgets.QLabel("Material Label:")
rec_label.setStyleSheet("font-size: 13px; font-weight: bold;")
rec_hl.addWidget(rec_label)

# Text input
label_input = QtWidgets.QLineEdit()
label_input.setPlaceholderText("e.g. asphalt, concrete, dirt...")
label_input.setFixedWidth(280)
rec_hl.addWidget(label_input)

# Max samples input
max_lbl = QtWidgets.QLabel("Max Samples:")
max_lbl.setStyleSheet("font-size: 13px;")
rec_hl.addWidget(max_lbl)

max_input = QtWidgets.QLineEdit()
max_input.setPlaceholderText("∞ unlimited")
max_input.setFixedWidth(110)
max_input.setStyleSheet("""
    background-color: #333333; color: #ffffff;
    border: 1px solid #555555; border-radius: 4px;
    padding: 4px 8px; font-size: 13px;
""")
rec_hl.addWidget(max_input)

# Start button
start_btn = QtWidgets.QPushButton("⏺  Start Recording")
start_btn.setObjectName("startBtn")
start_btn.setFixedWidth(180)
rec_hl.addWidget(start_btn)

# Stop button
stop_btn = QtWidgets.QPushButton("⏹  Stop")
stop_btn.setObjectName("stopBtn")
stop_btn.setFixedWidth(120)
stop_btn.setEnabled(False)
rec_hl.addWidget(stop_btn)

rec_hl.addSpacing(20)

# ---- ML PREDICTION DISPLAY ----
pred_frame = QtWidgets.QFrame()
pred_frame.setStyleSheet("border: 1px solid #555; border-radius: 6px; background: #1a1a2e; padding: 2px;")
pred_hl = QtWidgets.QHBoxLayout(pred_frame)
pred_hl.setContentsMargins(10, 4, 10, 4)
pred_hl.setSpacing(10)

pred_title = QtWidgets.QLabel("🔬 Surface:")
pred_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #aaaaaa; border: none;")
pred_hl.addWidget(pred_title)

pred_label = QtWidgets.QLabel("—")
pred_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff; border: none;")
pred_hl.addWidget(pred_label)

pred_conf = QtWidgets.QLabel("")
pred_conf.setStyleSheet("font-size: 11px; color: #888888; border: none;")
pred_hl.addWidget(pred_conf)

rec_hl.addWidget(pred_frame)
status_lbl = QtWidgets.QLabel("● Not recording")
status_lbl.setStyleSheet("font-size: 13px; color: #777777; padding: 4px 12px;")
rec_hl.addWidget(status_lbl)

rec_hl.addStretch()

# Sample counter
count_lbl = QtWidgets.QLabel("Samples: 0")
count_lbl.setStyleSheet("font-size: 13px; color: #aaaaaa;")
rec_hl.addWidget(count_lbl)

rec_hl.addSpacing(20)

# File info
file_lbl = QtWidgets.QLabel("No file open")
file_lbl.setStyleSheet("font-size: 11px; color: #666666;")
rec_hl.addWidget(file_lbl)

main_vl.addWidget(record_frame)

# ── Recording logic ────────────────────────────────────────────────────────────
def start_recording():
    global recording, csv_writer, csv_file, sample_count

    label = label_input.text().strip()
    if not label:
        status_lbl.setText("⚠  Enter a material label first!")
        status_lbl.setStyleSheet("font-size: 13px; color: #ffaa00; padding: 4px 12px;")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAVE_DIR, f"{label}_{timestamp}.csv")

    # Build CSV header
    header = ["label", "timestamp"]
    # Individual sensor columns
    for s in range(NUM_SENSORS):
        for wl in WAVELENGTHS:
            header.append(f"S{s}_{wl}nm")
    # Averaged columns
    for wl in WAVELENGTHS:
        header.append(f"AVG_{wl}nm")

    max_samples = 0  # 0 = unlimited
    try:
        val = int(max_input.text().strip())
        if val > 0:
            max_samples = val
    except ValueError:
        pass

    csv_file   = open(filename, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)
    sample_count = 0

    # Store max on the button for access in update()
    start_btn._max_samples = max_samples

    with record_lock:
        recording = True

    start_btn.setEnabled(False)
    stop_btn.setEnabled(True)
    label_input.setEnabled(False)
    max_input.setEnabled(False)
    limit_str = f" (max {max_samples})" if max_samples else " (unlimited)"
    status_lbl.setText(f"⏺  Recording: {label}{limit_str}")
    status_lbl.setStyleSheet("font-size: 13px; color: #ff4444; font-weight: bold; padding: 4px 12px;")
    file_lbl.setText(f"→ {os.path.basename(filename)}")
    count_lbl.setText("Samples: 0")

def stop_recording():
    global recording, csv_writer, csv_file

    with record_lock:
        recording = False

    if csv_file:
        csv_file.close()
        csv_file   = None
        csv_writer = None

    start_btn.setEnabled(True)
    stop_btn.setEnabled(False)
    label_input.setEnabled(True)
    max_input.setEnabled(True)
    status_lbl.setText(f"✔  Saved {sample_count} samples")
    status_lbl.setStyleSheet("font-size: 13px; color: #44cc44; padding: 4px 12px;")

start_btn.clicked.connect(start_recording)
stop_btn.clicked.connect(stop_recording)

# ── UPDATE FUNCTION ────────────────────────────────────────────────────────────
def update():
    global sample_count

    with lock:
        data = [row[:] for row in latest]

    for i in range(NUM_SENSORS):
        for j in range(N):
            smooth[i][j] += (data[i][j] - smooth[i][j]) * LERP

        peak_bar = max(max(smooth[i]) * 1.2, 100)
        for j in range(N):
            bar_items[i][j].setOpts(height=smooth[i][j])
        bar_plots[i].setYRange(0, peak_bar, padding=0)

        peak_line = max(max(hist[i][j]) for j in range(N))
        for j in range(N):
            line_items[i][j].setData(x_line, np.array(hist[i][j]))
        line_plots[i].setYRange(0, max(peak_line * 1.2, 100), padding=0)

        r, g, b = spectral_to_rgb(data[i])
        swatch_labels[i].setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #555; border-radius: 4px;")
        rgb_labels[i].setText(f"rgb({r},{g},{b})")

        header_row = "  ".join(f"{str(WAVELENGTHS[j]):>5}" for j in range(N))
        vals_row   = "  ".join(f"{data[i][j]:>5}" for j in range(N))
        value_labels[i].setText(f"{header_row}\n{vals_row}")

    avg = [sum(data[s][j] for s in range(NUM_SENSORS)) / NUM_SENSORS for j in range(N)]
    for j in range(N):
        hist_avg[j].append(avg[j])
        bis_avg[j].setOpts(height=avg[j])
        lis_avg[j].setData(x_line, np.array(hist_avg[j]))

    peak_avg_bar = max(max(avg) * 1.2, 100)
    bp_avg.setYRange(0, peak_avg_bar, padding=0)
    peak_avg_line = max(max(hist_avg[j]) for j in range(N))
    lp_avg.setYRange(0, max(peak_avg_line * 1.2, 100), padding=0)

    r, g, b = spectral_to_rgb([int(v) for v in avg])
    avg_swatch.setStyleSheet(
        f"background-color: rgb({r},{g},{b}); border: 2px solid #555; border-radius: 8px;")
    avg_rgb_label.setText(f"rgb({r}, {g}, {b})")
    avg_lines = "\n".join(f"{WAVELENGTHS[j]:>3}nm: {avg[j]:7.1f}" for j in range(N))
    avg_vals_label.setText(avg_lines)

    # ---- ML INFERENCE ----
    if ML_READY:
        try:
            features = np.array(avg).reshape(1, -1)
            pred_idx  = _model.predict(features)[0]
            proba     = _model.predict_proba(features)[0]
            pred_name = _le.inverse_transform([pred_idx])[0]
            confidence = proba[pred_idx] * 100

            # Color code by class
            colors_map = {
                'Asphalt':  '#444444',
                'Concrete': '#aaaaaa',
                'Brick':    '#cc6644',
            }
            color = colors_map.get(pred_name, '#ffffff')
            pred_label.setText(pred_name)
            pred_label.setStyleSheet(
                f"font-size: 18px; font-weight: bold; color: {color}; border: none;")

            # Show top 3 probabilities
            class_probs = sorted(zip(_le.classes_, proba), key=lambda x: -x[1])
            prob_str = "   ".join(f"{c}: {p*100:.0f}%" for c, p in class_probs)
            pred_conf.setText(prob_str)
        except Exception:
            pass

    # ── Save to CSV if recording ───────────────────────────────────────────────
    with record_lock:
        is_recording = recording

    if is_recording and csv_writer:
        ts  = datetime.datetime.now().isoformat()
        lbl = label_input.text().strip()
        row = [lbl, ts]
        for s in range(NUM_SENSORS):
            row.extend(data[s])
        row.extend([round(v, 2) for v in avg])
        csv_writer.writerow(row)
        csv_file.flush()
        sample_count += 1
        count_lbl.setText(f"Samples: {sample_count}")

        # Auto-stop if max samples reached
        max_s = getattr(start_btn, '_max_samples', 0)
        if max_s > 0 and sample_count >= max_s:
            stop_recording()

# ---- TIMER ----
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

win.show()
sys.exit(app.exec_())
