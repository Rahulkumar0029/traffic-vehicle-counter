"""Entry / exit counting across a gate line segment A -> B.

Fixes over the old counter: the vehicle must cross the actual SEGMENT (not the
infinite line), any line orientation works, direction is configurable, a
cooldown stops jitter double-counting, and stale tracks are purged.
"""
from typing import Dict, List

from .common import Detection
from .slots import Gate


def _side(a, b, p) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


class GateCounter:
    def __init__(self, gate: Gate, fps: float, cooldown_seconds: float = 2.0,
                 in_direction: str = "positive", anchor: str = "bottom"):
        if in_direction not in ("positive", "negative"):
            raise ValueError("in_direction must be 'positive' or 'negative'")
        self.gate = gate
        self.fps = float(fps)
        self.cooldown = max(1, round(cooldown_seconds * fps))
        self.in_direction = in_direction
        self.anchor = anchor
        self.prev: Dict[int, tuple] = {}
        self.seen: Dict[int, int] = {}
        self.last_cross: Dict[int, int] = {}
        self.counts = {"in": 0, "out": 0}
        self.by_class = {"in": {}, "out": {}}

    def _crosses_segment(self, p0, p1) -> bool:
        a, b = self.gate.p1, self.gate.p2
        s0, s1 = _side(a, b, p0), _side(a, b, p1)
        if s0 == 0 or s1 == 0 or (s0 > 0) == (s1 > 0):
            return False
        # movement segment must also straddle the gate line's endpoints
        d1, d2 = _side(p0, p1, a), _side(p0, p1, b)
        return (d1 > 0) != (d2 > 0)

    def update(self, detections: List[Detection], frame_idx: int) -> List[dict]:
        events = []
        for d in detections:
            cur = d.anchor(self.anchor)
            prev = self.prev.get(d.track_id)
            if prev is not None and self._crosses_segment(prev, cur):
                last = self.last_cross.get(d.track_id)
                if last is None or frame_idx - last >= self.cooldown:
                    s_prev = _side(self.gate.p1, self.gate.p2, prev)
                    going_positive = s_prev < 0  # negative -> positive side
                    is_in = going_positive == (self.in_direction == "positive")
                    key = "in" if is_in else "out"
                    self.counts[key] += 1
                    self.by_class[key][d.label] = self.by_class[key].get(d.label, 0) + 1
                    self.last_cross[d.track_id] = frame_idx
                    events.append({
                        "time_s": round(frame_idx / self.fps, 2),
                        "frame": frame_idx,
                        "event": f"vehicle_{key}",
                        "slot": "",
                        "track_id": d.track_id,
                        "label": d.label,
                        "duration_s": "",
                    })
            if _side(self.gate.p1, self.gate.p2, cur) != 0:
                self.prev[d.track_id] = cur  # keep last off-line point if exactly on it
            self.seen[d.track_id] = frame_idx

        if frame_idx % 300 == 0:  # purge tracks gone for > 5 s (no memory leak)
            stale = [t for t, f in self.seen.items() if frame_idx - f > 5 * self.fps]
            for t in stale:
                self.prev.pop(t, None)
                self.seen.pop(t, None)
                self.last_cross.pop(t, None)
        return events

    def in_arrow(self):
        """(start, end) of an arrow showing which way is IN, for drawing."""
        a, b = self.gate.p1, self.gate.p2
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        nx, ny = -(b[1] - a[1]), (b[0] - a[0])
        norm = (nx * nx + ny * ny) ** 0.5 or 1.0
        sign = 1 if self.in_direction == "positive" else -1
        L = 45
        return (int(mx), int(my)), (int(mx + sign * nx / norm * L),
                                    int(my + sign * ny / norm * L))
