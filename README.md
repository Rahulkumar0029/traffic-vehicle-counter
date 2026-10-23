# Smart Parking and Vehicle Detection & Tracking System

YOLOv8 + ByteTrack + OpenCV. Detects and tracks vehicles in a parking lot video,
decides which slots are free/occupied, counts vehicles entering/leaving through a
gate line, measures how long each vehicle parks, and logs everything.

## Setup
```
pip install -r requirements.txt
```
(`yolov8n.pt` downloads automatically on first run if it is not in the folder.)

## Use (3 steps)
```
# 1. Draw the slots (and optionally the gate line) on a frame of your video
python define_slots.py --source parking.mp4

# 2. Run
python main.py --source parking.mp4

# 3. Look in output/
```
Slot tool keys: click = add corner, **Enter** = finish slot, **r** = rectangle mode
(2 clicks), **g** = draw gate line (2 clicks, optional), **u** = undo, **s** = save,
**q** = quit. Press **Esc** in the main window to stop early.

Other inputs: `--source 0` (webcam), `--source rtsp://...`, `--no-display`,
`--max-frames 300`, `--model yolov8s.pt`.

## Outputs (`output/`)
| File | Content |
|---|---|
| `parking_out.mp4` | annotated video (green = free, red = occupied, parked timer, IN/OUT) |
| `events.csv`, `parking.db` | slot_occupied / slot_freed (with duration) / vehicle_in / vehicle_out |
| `occupancy_timeline.csv` | occupied/free count every second (plot this for the report) |
| `summary.json` | totals, peak occupancy, per-slot sessions, gate counts per class |

## How it works
1. YOLOv8 detects cars, motorcycles, buses, trucks; ByteTrack gives each a persistent ID.
2. Each vehicle's **bottom-centre point** is tested against every slot polygon.
3. A slot turns **occupied** only after a vehicle sits (almost) still inside it for
   `park_seconds`; it turns **free** after nothing is seen for `release_seconds`.
   Occupancy uses position, not track IDs, so ID switches cannot corrupt it.
4. Vehicles crossing the gate **segment** are counted IN or OUT (direction configurable).
5. Events are written to CSV/SQLite; the frame is annotated and saved.

All thresholds are in `config.yaml`.

## Tests
`python tests/test_logic.py` - checks occupancy, debouncing, gate counting and the full
pipeline using simulated detections (no YOLO needed).

## What changed from the old traffic counter
| Old | New |
|---|---|
| Count cars crossing a line on a road | Slot occupancy + entry/exit + dwell time |
| Only video files | Video file, webcam, RTSP |
| Class filter after inference, 6 classes | Filter inside YOLO (`classes=`), 4 vehicle classes |
| Line check ignored x-range | Segment crossing test, any orientation |
| Memory leak in `previous_centers` | Stale tracks purged |
| Per-class counts discarded | Reported in summary |
| No saved output | Video, CSV, SQLite, JSON summary |
| Hard-coded values | `config.yaml` |
| No tests, no requirements | Tests + `requirements.txt` |

## Legacy: traffic line counter
The original vehicle counting demo (which counts vehicles crossing a horizontal line on a road) has been preserved under [`legacy/traffic_counter/`](file:///c:/Users/rahul/OneDrive/Desktop/Projects/Traffic/legacy/traffic_counter/).

To run the legacy counter:
```bash
cd legacy/traffic_counter
python vehicle.py
```
Or from the repository root:
```bash
python legacy/traffic_counter/vehicle.py
```

