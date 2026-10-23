# Traffic Vehicle Counter & Predictor

An AI-powered computer vision system using YOLOv8 and OpenCV for real-time traffic vehicle detection, multi-object tracking, speed estimation, directional lane counting, and traffic volume forecasting.

---

## 🎯 Objectives Implemented

| Objective | Implementation Details |
|---|---|
| **1. Detect vehicles using AI** | YOLOv8 detects cars, trucks, buses, motorcycles, and bicycles with high precision. |
| **2. Track vehicles uniquely** | Persistent multi-object tracking assigns unique IDs to eliminate double-counting. |
| **3. Count vehicles automatically** | Categorizes vehicle flows into Outbound (southbound) and Inbound (northbound) lanes. |
| **4. Provide real-time monitoring** | Live HUD overlays: active vehicle counts, category breakdown, average speeds (px/s), dynamic trajectory trails, total vehicle counter, and real-time FPS counter. |
| **5. Enable future traffic prediction** | Aggregates traffic count time-series data and event logs to forecast future traffic volume and congestion levels. |

---

## 📊 Live Visual HUD Overview

The visual display matches the highway camera layout:

- **Highway Divider Line**: Blue vertical line dividing outbound (left roadway) and inbound (right roadway) traffic.
- **Top-Left (Outbound)**: Bright green counts and average speeds (e.g., `car: 21 (Avg: 218.9px/s)`).
- **Top-Left (Inbound)**: Blue counts and average speeds (e.g., `car: 9 (Avg: 218.9px/s)`).
- **Top-Right**: High-visibility bold red `TOTAL: <count>`.
- **Bottom-Right**: Real-time `FPS: <value>` in cyan.
- **Vehicle Markers**:
  - Green bounding box around detected vehicles.
  - Red dot at vehicle centroid.
  - Yellow label with confidence score (e.g., `car 0.7`).
  - Cyan motion vector trailing vehicle movement.

---

## 🚀 Quick Start

### 1. Run Live Video with Display Window

```powershell
py -3.11 vehicle.py
```
*(Or press `F5` in VS Code to launch directly.)*

Press `Esc` or `q` to exit the live window anytime.

### 2. Run in Headless Mode (Fast / Background)

```powershell
py -3.11 vehicle.py --no-display --max-frames 300
```

### 3. Generate Traffic Prediction & Forecast

```powershell
py -3.11 predict_traffic.py
```

---

## 📁 Output Artifacts Generated

All results are automatically generated in the `output/` directory:

- `output/traffic_screenshot.png`: High-resolution frame snapshot demonstrating detection and tracking HUD.
- `output/traffic_output.mp4`: Saved annotated video with full bounding boxes, trajectories, and statistics.
- `output/events.csv`: Per-vehicle log with timestamp, track ID, vehicle class, direction, confidence, and speed.
- `output/traffic_prediction_timeseries.csv`: Interval-based traffic counts formatted for time-series forecasting.
- `output/traffic_forecast.json`: Forecast report with hourly vehicle flow projection and congestion status.
- `output/summary.json`: Comprehensive session summary of vehicle categories and directional counts.

---

## ⚙️ Command-Line Options

```text
--video PATH          Path to input video (default: video.mp4)
--model PATH          Path to YOLO model (default: yolov8n.pt)
--output-dir PATH     Directory for outputs (default: output)
--max-frames N        Process N frames (0 for entire video)
--no-display          Disable OpenCV GUI window
--no-save-video       Skip saving annotated MP4 video
--conf FLOAT          YOLO confidence threshold (default: 0.25)
--divider-x INT       X-coordinate of median divider line (default: 635)
--trail-len INT       Length of trajectory trails (default: 15)
```
