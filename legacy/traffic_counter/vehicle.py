from argparse import ArgumentParser
from pathlib import Path

import cv2
from ultralytics import YOLO


VEHICLE_CLASSES = {"car", "bus", "truck", "motorbike", "motorcycle", "bicycle"}


def parse_args():
    parser = ArgumentParser(description="Count vehicles crossing a line in a video.")
    parser.add_argument("--video", type=Path, default=Path("video.mp4"))
    parser.add_argument("--model", type=Path, default=Path("../../yolov8n.pt"))
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    video_path = args.video if args.video.is_absolute() else base_dir / args.video
    model_path = args.model if args.model.is_absolute() else base_dir / args.model

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    count_line_position = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.76)
    inbound_count = 0
    outbound_count = 0
    inbound_counts = {}
    outbound_counts = {}
    counted_ids = set()
    previous_centers = {}
    frame_number = 0
    window_name = "Vehicle Detection & Counting"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            results = model.track(frame, persist=True, verbose=False)[0]
            frame_height, frame_width = frame.shape[:2]
            cv2.line(frame, (25, count_line_position), (frame_width - 25, count_line_position), (255, 127, 0), 3)

            if results.boxes.id is not None:
                for box, raw_track_id in zip(results.boxes, results.boxes.id):
                    track_id = int(raw_track_id.item())
                    cls_id = int(box.cls[0].item())
                    label = model.names[cls_id]
                    if label not in VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    previous_y = previous_centers.get(track_id)
                    direction = None
                    if previous_y is not None:
                        if previous_y < count_line_position <= center_y:
                            direction = "outbound"
                        elif previous_y > count_line_position >= center_y:
                            direction = "inbound"

                    if track_id not in counted_ids and direction is not None:
                        counted_ids.add(track_id)
                        if direction == "outbound":
                            outbound_count += 1
                            outbound_counts[label] = outbound_counts.get(label, 0) + 1
                        else:
                            inbound_count += 1
                            inbound_counts[label] = inbound_counts.get(label, 0) + 1
                        total_count = inbound_count + outbound_count
                        print(f"Vehicle counted ({direction}, {label}): {total_count}")

                    previous_centers[track_id] = center_y
                    box_color = (0, 255, 255) if track_id in counted_ids else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"{label} #{track_id}", (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            total_count = inbound_count + outbound_count
            cv2.putText(frame, f"INBOUND: {inbound_count}", (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 0), 2)
            cv2.putText(frame, f"OUTBOUND: {outbound_count}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2)
            cv2.putText(frame, f"TOTAL: {total_count}", (30, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

            if not args.no_display:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if args.max_frames and frame_number >= args.max_frames:
                break
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    total_count = inbound_count + outbound_count
    print(f"Inbound vehicles: {inbound_count}")
    print(f"Outbound vehicles: {outbound_count}")
    print(f"Total vehicles: {total_count}")
    print(f"Finished: {frame_number} frames processed.")


if __name__ == "__main__":
    main()
