from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Detection:
    """One tracked vehicle in one frame."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    label: str

    def anchor(self, mode: str = "bottom") -> Tuple[float, float]:
        """Point used for slot / gate tests.

        'bottom' = bottom-centre of the box (where the car touches the ground,
        best for slot tests on an angled camera). 'center' = box centre.
        """
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) / 2.0
        if mode == "center":
            return (cx, (y1 + y2) / 2.0)
        return (cx, float(y2))
