# Traffic Vehicle Counter

A YOLOv8 and OpenCV vehicle detector that tracks vehicles in video and counts them by movement direction.

## Features

- Detects cars, buses, trucks, motorcycles, and bicycles.
- Assigns tracking IDs to detected vehicles.
- Prevents the same tracked vehicle from being counted twice.
- Counts inbound and outbound vehicles separately.
- Displays the combined total count in real time.
- Supports recorded video input.
- Can run with or without a display window.

## How Direction Is Calculated

The program uses a horizontal counting line near the bottom of the video:

- A vehicle moving downward across the line is counted as **outbound**.
- A vehicle moving upward across the line is counted as **inbound**.
- Each tracking ID is counted only once.

These directions are based on the camera view. If the camera is repositioned, the meaning of inbound and outbound may need to be changed.

## Requirements

- Windows, macOS, or Linux
- Python 3.11 recommended
- The YOLOv8 Nano model included in this repository
- A video file named `video.mp4`

## Installation

Clone the repository:

```bash
git clone https://github.com/Rahulkumar0029/traffic-vehicle-counter.git
cd traffic-vehicle-counter
```

Install the Python packages:

```bash
python -m pip install ultralytics opencv-python
```

On Windows, if multiple Python versions are installed:

```powershell
py -3.11 -m pip install ultralytics opencv-python
```

## Run With Video Output

This opens a window showing the detections, tracking boxes, counting line, inbound count, outbound count, and total count:

```powershell
py -3.11 vehicle.py
```

Press `Esc` to stop the program.

## Run Without a Display

Use headless mode when running through a server, terminal, or cloud environment:

```powershell
py -3.11 vehicle.py --no-display
```

The final summary is printed in the terminal, for example:

```text
Inbound vehicles: 3
Outbound vehicles: 8
Total vehicles: 11
Finished: 300 frames processed.
```

## Command Options

```text
--video PATH       Video input path. Default: video.mp4
--model PATH       YOLO model path. Default: yolov8n.pt
--no-display       Disable the OpenCV display window
--max-frames N     Stop after N frames. Useful for testing
```

Example test run:

```powershell
py -3.11 vehicle.py --max-frames 300
```

## Project Files

- `vehicle.py` - Detection, tracking, directional counting, and display logic.
- `yolov8n.pt` - YOLOv8 Nano model used for vehicle detection.
- `video.mp4` - Sample traffic video.

## Technical Workflow

1. Load the YOLOv8 Nano model.
2. Open the input video with OpenCV.
3. Detect and track objects in each frame.
4. Filter detections to vehicle classes.
5. Calculate each vehicle's center point.
6. Detect when a tracked vehicle crosses the counting line.
7. Classify the movement as inbound or outbound.
8. Count each tracking ID once.
9. Display and print the directional totals.

## Current Scope

The current implementation estimates direction from the movement of a vehicle across a horizontal line. It does not currently provide calibrated speed in km/h, CSV logging, a browser dashboard, or live camera deployment.

## Repository

https://github.com/Rahulkumar0029/traffic-vehicle-counter
