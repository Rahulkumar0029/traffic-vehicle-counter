"""Smart Parking and Vehicle Detection & Tracking System.

python main.py --source parking.mp4
python main.py --source 0                       (webcam)
python main.py --source rtsp://...  --no-display
"""
import argparse
import json
from pathlib import Path

import yaml

from parking.detector import YoloDetector
from parking.pipeline import run
from parking.slots import load_layout

BASE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Smart parking: slot occupancy, "
                                "vehicle detection, tracking and entry/exit counting.")
    p.add_argument("--source", required=True,
                   help="video file, webcam index (0) or RTSP/HTTP URL")
    p.add_argument("--config", default=str(BASE / "config.yaml"))
    p.add_argument("--slots", help="slots.json (overrides config)")
    p.add_argument("--model", help="YOLO weights (overrides config)")
    p.add_argument("--no-display", action="store_true", help="no preview window")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    slots_file = Path(args.slots or cfg.get("slots_file", "slots.json"))
    if not slots_file.is_absolute() and not slots_file.exists():
        slots_file = BASE / slots_file
    slots, gate = load_layout(slots_file)

    detector = YoloDetector(
        model=args.model or cfg.get("model", "yolov8n.pt"),
        conf=cfg.get("conf", 0.35), imgsz=cfg.get("imgsz", 640),
        classes=cfg.get("classes", [2, 3, 5, 7]),
        tracker=cfg.get("tracker", "bytetrack.yaml"))

    summary = run(args.source, detector, slots, gate, cfg,
                  display=cfg.get("display", True) and not args.no_display,
                  max_frames=args.max_frames)
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
