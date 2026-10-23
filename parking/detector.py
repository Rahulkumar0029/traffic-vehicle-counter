"""YOLOv8 detection + tracking wrapper (the only file that needs ultralytics)."""
from pathlib import Path
from typing import List

from .common import Detection


class YoloDetector:
    def __init__(self, model="yolov8n.pt", conf=0.35, imgsz=640,
                 classes=(2, 3, 5, 7), tracker="bytetrack.yaml"):
        from ultralytics import YOLO  # imported here so tests run without it
        # If the file exists use it, otherwise let ultralytics download by name.
        self.model = YOLO(str(model) if Path(model).is_file() else str(model))
        self.conf = conf
        self.imgsz = imgsz
        self.classes = list(classes)
        self.tracker = tracker

    def __call__(self, frame) -> List[Detection]:
        res = self.model.track(
            frame, persist=True, conf=self.conf, imgsz=self.imgsz,
            classes=self.classes, tracker=self.tracker, verbose=False)[0]
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().tolist()
        cls = boxes.cls.int().cpu().tolist()
        return [
            Detection(int(i), tuple(float(v) for v in b), self.model.names[int(c)])
            for i, b, c in zip(ids, xyxy, cls)
        ]
