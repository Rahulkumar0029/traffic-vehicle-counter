"""Slot occupancy engine.

A slot becomes OCCUPIED only when a vehicle has been inside it, almost
motionless, for `park_seconds`. This ignores cars that merely drive past.
A slot becomes FREE only after no vehicle has been seen in it for
`release_seconds`, which hides single-frame detector misses.

Occupancy is decided from POSITION, not from tracker IDs, so tracker ID
switches cannot corrupt the slot status.
"""
from collections import deque
from typing import Dict, List

from .common import Detection
from .slots import Slot


class _SlotState:
    def __init__(self, slot: Slot, window: int):
        self.slot = slot
        self.history = deque(maxlen=window)  # anchor point or None per frame
        self.occupied = False
        self.absent_frames = 0
        self.arrival_frame = None
        self.last_seen_frame = None
        self.label = ""
        self.track_id = None
        self.sessions = 0
        self.total_occupied_s = 0.0


class OccupancyEngine:
    def __init__(self, slots: List[Slot], fps: float, frame_width: int,
                 park_seconds: float = 2.0, release_seconds: float = 3.0,
                 min_presence: float = 0.8, move_threshold_frac: float = 0.02,
                 anchor: str = "bottom"):
        self.fps = float(fps)
        self.window = max(2, round(park_seconds * fps))
        self.release_frames = max(1, round(release_seconds * fps))
        self.min_presence = min_presence
        self.move_threshold = move_threshold_frac * frame_width
        self.anchor = anchor
        self.states: Dict[str, _SlotState] = {
            s.id: _SlotState(s, self.window) for s in slots
        }
        self.peak_occupied = 0

    # ------------------------------------------------------------------
    def _vehicle_in_slot(self, slot: Slot, detections: List[Detection]):
        best, best_d = None, float("inf")
        for d in detections:
            p = d.anchor(self.anchor)
            if slot.contains(p):
                dist = (p[0] - slot.cx) ** 2 + (p[1] - slot.cy) ** 2
                if dist < best_d:
                    best, best_d = d, dist
        return best

    def _is_parked(self, st: _SlotState) -> bool:
        if len(st.history) < self.window:
            return False
        pts = [p for p in st.history if p is not None]
        if len(pts) < self.min_presence * self.window:
            return False
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        spread = max(max(xs) - min(xs), max(ys) - min(ys))
        return spread <= self.move_threshold

    # ------------------------------------------------------------------
    def update(self, detections: List[Detection], frame_idx: int) -> List[dict]:
        events = []
        for st in self.states.values():
            veh = self._vehicle_in_slot(st.slot, detections)
            st.history.append(veh.anchor(self.anchor) if veh else None)

            if not st.occupied:
                if self._is_parked(st):
                    st.occupied = True
                    st.absent_frames = 0
                    st.arrival_frame = frame_idx - self.window + 1
                    st.last_seen_frame = frame_idx if veh else frame_idx - 1
                    st.label = veh.label if veh else ""
                    st.track_id = veh.track_id if veh else None
                    events.append({
                        "time_s": round(frame_idx / self.fps, 2),
                        "frame": frame_idx,
                        "event": "slot_occupied",
                        "slot": st.slot.id,
                        "track_id": st.track_id if st.track_id is not None else "",
                        "label": st.label,
                        "duration_s": "",
                    })
            else:
                if veh:
                    st.absent_frames = 0
                    st.last_seen_frame = frame_idx
                    st.label = veh.label
                    st.track_id = veh.track_id
                else:
                    st.absent_frames += 1
                    if st.absent_frames >= self.release_frames:
                        duration = max(
                            0.0, (st.last_seen_frame - st.arrival_frame) / self.fps)
                        st.occupied = False
                        st.sessions += 1
                        st.total_occupied_s += duration
                        events.append({
                            "time_s": round(frame_idx / self.fps, 2),
                            "frame": frame_idx,
                            "event": "slot_freed",
                            "slot": st.slot.id,
                            "track_id": st.track_id if st.track_id is not None else "",
                            "label": st.label,
                            "duration_s": round(duration, 2),
                        })
                        st.history.clear()
                        st.arrival_frame = None

        self.peak_occupied = max(self.peak_occupied, self.occupied_count())
        return events

    # ------------------------------------------------------------------
    def occupied_count(self) -> int:
        return sum(1 for s in self.states.values() if s.occupied)

    def total(self) -> int:
        return len(self.states)

    def free_count(self) -> int:
        return self.total() - self.occupied_count()

    def snapshot(self, frame_idx: int):
        """Per-slot info for drawing."""
        out = []
        for st in self.states.values():
            parked_s = None
            if st.occupied and st.arrival_frame is not None:
                parked_s = max(0.0, (frame_idx - st.arrival_frame) / self.fps)
            out.append({"id": st.slot.id, "polygon": st.slot.int_polygon(),
                        "occupied": st.occupied, "parked_s": parked_s,
                        "cx": st.slot.cx, "cy": st.slot.cy})
        return out

    def slot_stats(self, final_frame: int):
        """Finish open sessions (vehicle still parked at end of video)."""
        stats = []
        for st in self.states.values():
            sessions, total = st.sessions, st.total_occupied_s
            if st.occupied and st.arrival_frame is not None:
                sessions += 1
                total += max(0.0, (final_frame - st.arrival_frame) / self.fps)
            stats.append({"slot": st.slot.id,
                          "final_state": "occupied" if st.occupied else "free",
                          "sessions": sessions,
                          "total_occupied_s": round(total, 1)})
        return stats
