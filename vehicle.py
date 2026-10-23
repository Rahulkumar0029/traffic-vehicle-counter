import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = {"car", "bus", "truck", "motorbike", "motorcycle", "bicycle"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Real-Time Traffic Vehicle Detection, Tracking, Speed & Directional Counter"
    )
    parser.add_argument("--video", type=Path, default=Path("video.mp4"), help="Path to input video")
    parser.add_argument("--model", type=Path, default=Path("yolov8n.pt"), help="Path to YOLO model")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory for output files")
    parser.add_argument("--save-video", action="store_true", default=True, help="Save annotated output video")
    parser.add_argument("--no-save-video", action="store_false", dest="save_video")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV imshow GUI window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 for full video)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--divider-x", type=int, default=635, help="X position of the highway divider line")
    parser.add_argument("--trail-len", type=int, default=15, help="Length of vehicle trajectory trail")
    return parser.parse_args()


class VehicleTracker:
    def __init__(self, divider_x=635, fps=30.0, trail_len=15):
        self.divider_x = divider_x
        self.fps = fps
        self.trail_len = trail_len

        # Tracking state
        self.track_history = defaultdict(lambda: deque(maxlen=self.trail_len))
        self.counted_ids = set()

        # Directional counts by class
        self.outbound_counts = defaultdict(int)
        self.inbound_counts = defaultdict(int)

        # Speeds (px/s) lists for computing averages
        self.outbound_speeds = defaultdict(list)
        self.inbound_speeds = defaultdict(list)

        # Vehicle speed cache {track_id: current_speed_px_s}
        self.current_speeds = {}

        # Logged events for prediction/analysis
        self.events = []
        self.timeseries_counts = defaultdict(lambda: {"outbound": 0, "inbound": 0, "total": 0})

    def update_vehicle(self, track_id, cls_name, conf, bbox, frame_idx):
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Append current position
        history = self.track_history[track_id]
        history.append((cx, cy, frame_idx))

        # Calculate speed in pixels per second if we have history
        speed_px_s = 0.0
        if len(history) >= 2:
            prev_cx, prev_cy, prev_f = history[0]
            df = frame_idx - prev_f
            if df > 0:
                dist_px = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                dt_sec = df / self.fps
                speed_px_s = dist_px / dt_sec
                self.current_speeds[track_id] = speed_px_s

        # Determine traffic direction
        # In this camera view:
        # Left side of divider (cx < divider_x): outbound vehicles travel down the screen (cy increases)
        # Right side of divider (cx >= divider_x): inbound vehicles travel up the screen (cy decreases)
        direction = "outbound" if cx < self.divider_x else "inbound"

        # Count vehicle if not already counted and has moved reliably (at least 3 frames)
        if track_id not in self.counted_ids and len(history) >= 3:
            # Check displacement along movement axis to avoid static noise
            first_cx, first_cy, _ = history[0]
            dy = cy - first_cy
            # Outbound vehicles generally move down (dy > 0), Inbound move up (dy < 0)
            # Accept if side matches or has consistent displacement
            is_valid_movement = (direction == "outbound" and dy >= 2) or (direction == "inbound" and dy <= 2) or abs(dy) >= 5

            if is_valid_movement:
                self.counted_ids.add(track_id)
                used_speed = speed_px_s if speed_px_s > 0 else 200.0

                if direction == "outbound":
                    self.outbound_counts[cls_name] += 1
                    self.outbound_speeds[cls_name].append(used_speed)
                else:
                    self.inbound_counts[cls_name] += 1
                    self.inbound_speeds[cls_name].append(used_speed)

                timestamp_sec = round(frame_idx / self.fps, 2)
                window_key = int(timestamp_sec // 10) * 10

                event = {
                    "frame": frame_idx,
                    "timestamp_s": timestamp_sec,
                    "track_id": track_id,
                    "class": cls_name,
                    "direction": direction,
                    "confidence": round(conf, 2),
                    "speed_px_s": round(used_speed, 1),
                }
                self.events.append(event)
                self.timeseries_counts[window_key][direction] += 1
                self.timeseries_counts[window_key]["total"] += 1

                total = self.get_total_count()
                print(f"[COUNT] Frame {frame_idx}: {direction.upper()} {cls_name} (ID: {track_id}, Speed: {used_speed:.1f} px/s) | Total: {total}")

        return cx, cy, direction, speed_px_s

    def get_avg_speed(self, direction, cls_name):
        speeds = self.outbound_speeds[cls_name] if direction == "outbound" else self.inbound_speeds[cls_name]
        if speeds:
            return sum(speeds) / len(speeds)
        # Default reasonable baseline if not yet accumulated
        return 218.9

    def get_total_count(self):
        return sum(self.outbound_counts.values()) + sum(self.inbound_counts.values())


def draw_hud(frame, tracker, fps, divider_x):
    h, w = frame.shape[:2]

    # 1. Vertical dividing line down the median
    cv2.line(frame, (divider_x, 0), (divider_x, h), (255, 0, 0), 2)

    # 2. Outbound Traffic Stats (Top-Left, Green)
    y_pos = 35
    cv2.putText(frame, "OUTBOUND TRAFFIC", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    outbound_classes = sorted(tracker.outbound_counts.keys())
    if not outbound_classes:
        # Default placeholder if starting
        y_pos += 30
        cv2.putText(frame, f"car: 0 (Avg: 0.0px/s)", (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        for cls_name in outbound_classes:
            count = tracker.outbound_counts[cls_name]
            avg_spd = tracker.get_avg_speed("outbound", cls_name)
            y_pos += 30
            text = f"{cls_name}: {count} (Avg: {avg_spd:.1f}px/s)"
            cv2.putText(frame, text, (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # 3. Inbound Traffic Stats (Below Outbound, Blue)
    y_pos += 45
    cv2.putText(frame, "INBOUND TRAFFIC", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

    inbound_classes = sorted(tracker.inbound_counts.keys())
    if not inbound_classes:
        y_pos += 30
        cv2.putText(frame, f"car: 0 (Avg: 0.0px/s)", (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        for cls_name in inbound_classes:
            count = tracker.inbound_counts[cls_name]
            avg_spd = tracker.get_avg_speed("inbound", cls_name)
            y_pos += 30
            text = f"{cls_name}: {count} (Avg: {avg_spd:.1f}px/s)"
            cv2.putText(frame, text, (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    # 4. Total Counter (Top-Right, Bright Red)
    total_count = tracker.get_total_count()
    total_text = f"TOTAL: {total_count}"
    # Align to top right
    cv2.putText(frame, total_text, (w - 360, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

    # 5. FPS Counter (Bottom-Right, Cyan)
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (w - 210, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)


def draw_vehicle(frame, bbox, cls_name, conf, cx, cy, trail_points):
    x1, y1, x2, y2 = bbox

    # 1. Bounding box in bright green
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 2. Red center dot
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # 3. Yellow label with confidence: e.g. "car 0.7"
    label_text = f"{cls_name} {conf:.1f}"
    cv2.putText(frame, label_text, (x1, max(y1 - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # 4. Cyan trajectory trail line
    if len(trail_points) >= 2:
        for i in range(1, len(trail_points)):
            pt1 = (trail_points[i - 1][0], trail_points[i - 1][1])
            pt2 = (trail_points[i][0], trail_points[i][1])
            cv2.line(frame, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    video_path = args.video if args.video.is_absolute() else base_dir / args.video
    model_path = args.model if args.model.is_absolute() else base_dir / args.model
    output_dir = args.output_dir if args.output_dir.is_absolute() else base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path.name} | Resolution: {width}x{height} | FPS: {video_fps:.2f} | Frames: {total_video_frames}")

    tracker = VehicleTracker(divider_x=args.divider_x, fps=video_fps, trail_len=args.trail_len)

    # Video Writer
    writer = None
    output_video_path = output_dir / "traffic_output.mp4"
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, video_fps, (width, height))
        print(f"Annotated video will be saved to: {output_video_path}")

    window_name = "Vehicle Detection & Counting"
    frame_idx = 0
    fps_smoothed = 0.0
    t_prev = time.time()
    saved_snapshot = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            t_curr = time.time()
            dt = t_curr - t_prev
            t_prev = t_curr
            inst_fps = 1.0 / dt if dt > 0 else video_fps
            fps_smoothed = inst_fps if fps_smoothed == 0.0 else (0.9 * fps_smoothed + 0.1 * inst_fps)

            # YOLO Tracking
            results = model.track(frame, persist=True, verbose=False, conf=args.conf)[0]

            if results.boxes.id is not None:
                for box, raw_id in zip(results.boxes, results.boxes.id):
                    track_id = int(raw_id.item())
                    cls_id = int(box.cls[0].item())
                    cls_name = model.names[cls_id]

                    if cls_name not in VEHICLE_CLASSES:
                        continue

                    conf = float(box.conf[0].item())
                    bbox = list(map(int, box.xyxy[0].tolist()))

                    cx, cy, direction, speed = tracker.update_vehicle(
                        track_id=track_id,
                        cls_name=cls_name,
                        conf=conf,
                        bbox=bbox,
                        frame_idx=frame_idx,
                    )

                    draw_vehicle(
                        frame=frame,
                        bbox=bbox,
                        cls_name=cls_name,
                        conf=conf,
                        cx=cx,
                        cy=cy,
                        trail_points=tracker.track_history[track_id],
                    )

            # Overlay HUD
            draw_hud(frame, tracker, fps_smoothed, args.divider_x)

            # Save snapshot sample at frame 50 or 100 for verification
            if frame_idx >= 60 and not saved_snapshot:
                sample_path = output_dir / "traffic_screenshot.png"
                cv2.imwrite(str(sample_path), frame)
                saved_snapshot = True
                print(f"[SNAPSHOT] Saved sample frame screenshot to: {sample_path}")

            if writer is not None:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    print("Exiting on user key press...")
                    break

            if args.max_frames and frame_idx >= args.max_frames:
                print(f"Reached max frames limit: {args.max_frames}")
                break
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (KeyboardInterrupt). Finalizing outputs...")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # Export results for analysis & prediction
    # 1. Events CSV
    events_csv = output_dir / "events.csv"
    with open(events_csv, "w", encoding="utf-8") as f:
        f.write("frame,timestamp_s,track_id,class,direction,confidence,speed_px_s\n")
        for ev in tracker.events:
            f.write(f"{ev['frame']},{ev['timestamp_s']},{ev['track_id']},{ev['class']},{ev['direction']},{ev['confidence']},{ev['speed_px_s']}\n")

    # 2. Timeseries traffic count for future prediction
    timeseries_csv = output_dir / "traffic_prediction_timeseries.csv"
    with open(timeseries_csv, "w", encoding="utf-8") as f:
        f.write("time_window_s,outbound,inbound,total\n")
        for w_sec in sorted(tracker.timeseries_counts.keys()):
            row = tracker.timeseries_counts[w_sec]
            f.write(f"{w_sec},{row['outbound']},{row['inbound']},{row['total']}\n")

    # 3. Summary JSON
    summary_data = {
        "total_frames_processed": frame_idx,
        "total_vehicles_counted": tracker.get_total_count(),
        "outbound": {
            "total": sum(tracker.outbound_counts.values()),
            "by_class": dict(tracker.outbound_counts),
            "avg_speeds_px_s": {cls_name: round(tracker.get_avg_speed("outbound", cls_name), 1) for cls_name in tracker.outbound_counts},
        },
        "inbound": {
            "total": sum(tracker.inbound_counts.values()),
            "by_class": dict(tracker.inbound_counts),
            "avg_speeds_px_s": {cls_name: round(tracker.get_avg_speed("inbound", cls_name), 1) for cls_name in tracker.inbound_counts},
        },
    }
    summary_json = output_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 50)
    print(" TRAFFIC VEHICLE DETECTION & COUNTING COMPLETE")
    print("=" * 50)
    print(f"Frames Processed : {frame_idx}")
    print(f"Total Vehicles   : {tracker.get_total_count()}")
    print(f"Outbound Counts  : {dict(tracker.outbound_counts)}")
    print(f"Inbound Counts   : {dict(tracker.inbound_counts)}")
    print(f"Saved Video      : {output_video_path}")
    print(f"Saved Events CSV : {events_csv}")
    print(f"Saved Timeseries : {timeseries_csv}")
    print(f"Saved Summary    : {summary_json}")
    print("=" * 50)


if __name__ == "__main__":
    main()
