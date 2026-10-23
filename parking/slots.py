"""Parking-slot geometry and the slots.json layout file."""
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


class Slot:
    def __init__(self, slot_id: str, polygon):
        self.id = slot_id
        self.polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(self.polygon) < 3:
            raise ValueError(f"Slot {slot_id} needs at least 3 points")
        self._contour = self.polygon.reshape(-1, 1, 2)
        self.cx = float(self.polygon[:, 0].mean())
        self.cy = float(self.polygon[:, 1].mean())

    def contains(self, point: Tuple[float, float]) -> bool:
        return cv2.pointPolygonTest(
            self._contour, (float(point[0]), float(point[1])), False
        ) >= 0

    def int_polygon(self) -> np.ndarray:
        return self.polygon.astype(np.int32)


class Gate:
    """Entry/exit line defined by two points A -> B."""

    def __init__(self, p1, p2):
        self.p1 = (float(p1[0]), float(p1[1]))
        self.p2 = (float(p2[0]), float(p2[1]))


def load_layout(path) -> Tuple[List[Slot], Optional[Gate]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run:  python define_slots.py --source <video>"
        )
    data = json.loads(path.read_text())
    slots = [Slot(s["id"], s["polygon"]) for s in data.get("slots", [])]
    if not slots:
        raise ValueError(f"{path} contains no slots")
    gate = None
    g = data.get("gate")
    if g:
        gate = Gate(g["p1"], g["p2"])
    return slots, gate


def save_layout(path, slots_polygons, gate=None, image_size=None):
    """slots_polygons: list of (id, [[x, y], ...]); gate: ((x,y),(x,y)) or None."""
    data = {
        "image_size": list(image_size) if image_size else None,
        "slots": [
            {"id": sid, "polygon": [[int(x), int(y)] for x, y in poly]}
            for sid, poly in slots_polygons
        ],
        "gate": (
            {"p1": [int(gate[0][0]), int(gate[0][1])],
             "p2": [int(gate[1][0]), int(gate[1][1])]}
            if gate else None
        ),
    }
    Path(path).write_text(json.dumps(data, indent=2))
