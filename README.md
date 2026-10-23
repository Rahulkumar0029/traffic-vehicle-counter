# Traffic Vehicle Counter & Predictor

An AI-powered computer vision system built with **YOLOv8** and **OpenCV** for real-time highway vehicle detection, multi-object tracking, speed estimation, directional lane counting, and traffic volume forecasting.

---

## 🎯 Objectives Implemented

| Objective | Description & Implementation Details |
|---|---|
| **1. Detect vehicles using AI** | Leverages YOLOv8 to detect vehicles (`car`, `truck`, `bus`, `motorcycle`, `bicycle`) with high precision across varying distances. |
| **2. Track vehicles uniquely** | Persistent multi-object tracking assigns unique tracking IDs (`track_id`) across frames to prevent double-counting. |
| **3. Count vehicles automatically** | Classifies and tallies traffic flows into **OUTBOUND** (southbound / towards camera) and **INBOUND** (northbound / away from camera) streams. |
| **4. Provide real-time monitoring** | Live HUD overlays: lane-level counts, class breakdown, average speeds (`px/s`), dynamic cyan trajectory motion trails, red total counter, and real-time FPS counter. |
| **5. Enable future traffic prediction** | Automatically logs per-vehicle events and aggregated 10-second traffic count time-series to predict future traffic trends and congestion status. |

---

## 📊 Live Visual HUD Overview

The visual overlay matches real-world highway surveillance monitoring:

- **Highway Divider Line**: Blue vertical line (`x = 635`) separating the outbound carriageway (left) and inbound carriageway (right).
- **Top-Left (Outbound Traffic)**: Bright green header displaying active category counts and running average speed (e.g. `car: 21 (Avg: 218.9px/s)`).
- **Top-Left (Inbound Traffic)**: Blue header displaying active category counts and running average speed (e.g. `car: 9 (Avg: 218.9px/s)`).
- **Top-Right (Total Counter)**: High-visibility bold red `TOTAL: <count>`.
- **Bottom-Right (Performance)**: Cyan real-time `FPS: <value>`.
- **Vehicle Bounding Boxes & Tracking**:
  - Bright green bounding box around detected vehicles.
  - Centered solid red dot at the vehicle centroid `(cx, cy)`.
  - Yellow text displaying vehicle class and confidence score (e.g. `car 0.7`).
  - Dynamic cyan motion vector / trajectory trail showing vehicle movement history.

---

## 🏗️ Project Architecture & Files

```text
Traffic/
├── vehicle.py                   # Main detection, tracking, speed calculation & HUD counter
├── predict_traffic.py           # Traffic volume prediction & congestion forecasting model
├── requirements.txt             # Python project dependencies
├── README.md                    # Project documentation
├── yolov8n.pt                   # YOLOv8 Nano model weights
├── video.mp4                    # Sample highway traffic video (1280x720, 29.97 FPS)
├── .gitignore                   # Ignores virtual environment and heavy output media
├── .vscode/
│   └── launch.json              # VS Code run & debug configurations (F5 to launch)
└── output/                      # Generated automatically at runtime
    ├── traffic_screenshot.png   # Sample frame capture with full HUD
    ├── traffic_output.mp4       # Full annotated video with bounding boxes & metrics
    ├── events.csv               # Per-vehicle log (frame, time, id, class, direction, speed)
    ├── traffic_prediction_timeseries.csv  # 10s windowed vehicle counts
    ├── traffic_forecast.json    # Forecast report with projected hourly flow
    └── summary.json             # Session summary statistics
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10 or 3.11 recommended
- Git

### 2. Clone Repository
```bash
git clone https://github.com/Rahulkumar0029/traffic-vehicle-counter.git
cd traffic-vehicle-counter
```

### 3. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```
*(On Windows with multiple Python versions installed, use `py -3.11 -m pip install -r requirements.txt`)*

---

## 🚀 How to Run

### 1. Interactive Live GUI Mode (OpenCV Display)
Runs the live tracking window with video display and real-time HUD metrics:
```powershell
py -3.11 vehicle.py
```
> **Tip:** You can also press **`F5`** directly inside VS Code to launch the program.  
> Press **`Esc`** or **`q`** on your keyboard to stop the run cleanly at any time.

### 2. Fast / Headless Mode (Background Processing)
Runs without the GUI display window (ideal for servers or batch processing):
```powershell
py -3.11 vehicle.py --no-display
```

To run on a specific number of frames (e.g., 300 frames for testing):
```powershell
py -3.11 vehicle.py --no-display --max-frames 300
```

### 3. Generate Traffic Prediction & Congestion Forecast
Analyzes the generated traffic time-series data and produces a future volume forecast:
```powershell
py -3.11 predict_traffic.py
```

Sample output:
```text
==================================================
 TRAFFIC PREDICTION & FORECAST REPORT
==================================================
Windows Analyzed     : 30
Avg Vehicles / 10s   : 7.50
Projected Hourly Rate: 2700.0 vehicles/hour
Congestion Status    : Heavy / Congested
Trend Direction      : Stable (+0.12/window)
Forecasts Saved To   : output\traffic_forecast.json
==================================================
```

---

## 🛠️ Command-Line Options

### `vehicle.py` Options:
```text
--video PATH          Path to input video file (default: video.mp4)
--model PATH          Path to YOLOv8 model weights (default: yolov8n.pt)
--output-dir PATH     Output directory for logs and videos (default: output)
--max-frames N        Limit processing to N frames (default: 0 = full video)
--conf FLOAT          YOLO detection confidence threshold (default: 0.25)
--divider-x INT       X-coordinate of highway median divider line (default: 635)
--trail-len INT       Length of vehicle motion trajectory trail (default: 15)
--no-display          Disable OpenCV GUI display window
--no-save-video       Skip saving annotated MP4 output video
```

### `predict_traffic.py` Options:
```text
--timeseries PATH     Path to traffic time-series CSV (default: output/traffic_prediction_timeseries.csv)
--events PATH         Path to vehicle events CSV (default: output/events.csv)
--output PATH         Path for forecast JSON export (default: output/traffic_forecast.json)
--forecast-windows N  Number of upcoming 10-second windows to forecast (default: 5)
```

---

## 📁 Output Artifacts Explained

| File | Purpose |
|---|---|
| `output/traffic_screenshot.png` | Snapshot showing detection boxes, confidence, speeds, divider, and counters. |
| `output/traffic_output.mp4` | Recorded video with all HUD elements for presentation and verification. |
| `output/events.csv` | Detailed log of every counted vehicle (`frame`, `timestamp_s`, `track_id`, `class`, `direction`, `confidence`, `speed_px_s`). |
| `output/traffic_prediction_timeseries.csv` | Windowed traffic counts (`time_window_s`, `outbound`, `inbound`, `total`) for time-series modeling. |
| `output/traffic_forecast.json` | Traffic flow projection, average flow rate, congestion category, and trend. |
| `output/summary.json` | Final aggregate counts and average speeds per vehicle class and direction. |
