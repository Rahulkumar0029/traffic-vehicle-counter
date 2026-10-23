"""Run:  python tests/test_logic.py   (no YOLO needed)."""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parking.common import Detection
from parking.gate import GateCounter
from parking.occupancy import OccupancyEngine
from parking.pipeline import run
from parking.slots import Gate, Slot, load_layout, save_layout

FPS, W = 10, 640
SLOT = Slot("S1", [(200, 200), (300, 200), (300, 300), (200, 300)])


def car(x_center, y_bottom, tid=1, label="car", w=60, h=40):
    return Detection(tid, (x_center - w / 2, y_bottom - h, x_center + w / 2, y_bottom), label)


def engine(slots=None):
    return OccupancyEngine(slots or [SLOT], FPS, W, park_seconds=2, release_seconds=3)


def test_passing_car_does_not_occupy():
    e = engine()
    for f in range(80):                       # 30 px/frame through the slot
        e.update([car(100 + f * 30 % 600, 250)] if f < 40 else [], f)
    assert e.occupied_count() == 0
    e = engine()
    for f in range(80):                       # slow creep, 4 px/frame, never settles
        e.update([car(150 + f * 4, 250)], f)
    assert e.occupied_count() == 0, "slow mover must not count as parked"


def test_parked_car_occupies_once_and_survives_dropouts():
    e, evs = engine(), []
    for f in range(60):
        jitter = (f % 3) - 1                  # +-1 px detector jitter
        evs += e.update([car(250 + jitter, 290 + jitter)], f)
    assert e.occupied_count() == 1
    assert [x["event"] for x in evs] == ["slot_occupied"]
    for f in range(60, 70):                   # 1 s of missed detections (< release)
        evs += e.update([], f)
    assert e.occupied_count() == 1, "short dropout must not free the slot"
    for f in range(70, 120):
        evs += e.update([car(250, 290)], f)
    assert len([x for x in evs if x["event"] == "slot_occupied"]) == 1


def test_leaving_frees_slot_and_duration_is_right():
    e, evs = engine(), []
    for f in range(100):                      # parked frames 0..99
        evs += e.update([car(250, 290)], f)
    for f in range(100, 140):
        evs += e.update([], f)
    assert e.occupied_count() == 0
    freed = [x for x in evs if x["event"] == "slot_freed"]
    assert len(freed) == 1
    assert abs(freed[0]["duration_s"] - 9.9) < 0.2, freed[0]  # ~ (99-0)/10
    assert e.slot_stats(140)[0]["sessions"] == 1


def test_slots_are_independent_and_use_bottom_point():
    s2 = Slot("S2", [(320, 200), (420, 200), (420, 300), (320, 300)])
    e = engine([SLOT, s2])
    for f in range(40):
        e.update([car(250, 290), car(370, 290, tid=2)], f)
    assert e.occupied_count() == 2 and e.free_count() == 0
    # a tall box whose CENTRE is in the slot but whose bottom is below it
    e = engine()
    for f in range(40):
        e.update([Detection(1, (220, 250, 280, 340), "car")], f)
    assert e.occupied_count() == 0


GATE = Gate((100, 300), (500, 300))


def gate(direction="positive"):
    return GateCounter(GATE, FPS, cooldown_seconds=2, in_direction=direction)


def test_gate_counts_both_directions_and_ignores_outside_segment():
    g = gate()
    for f, y in enumerate([260, 280, 310, 330]):          # downward at x=300
        g.update([car(300, y)], f)
    assert g.counts == {"in": 1, "out": 0}
    for f, y in enumerate([330, 310, 290, 270], start=100):  # upward, other car
        g.update([car(300, y, tid=2)], f)
    assert g.counts == {"in": 1, "out": 1}
    for f, y in enumerate([260, 280, 310, 330], start=200):  # x=50, outside segment
        g.update([car(50, y, tid=3)], f)
    assert g.counts == {"in": 1, "out": 1}
    g2 = gate("negative")
    for f, y in enumerate([260, 310]):
        g2.update([car(300, y)], f)
    assert g2.counts == {"in": 0, "out": 1}


def test_gate_cooldown_stops_jitter_double_count():
    g = gate()
    for f, y in enumerate([298, 302, 298, 302, 298, 302]):  # wobbling on the line
        g.update([car(300, y)], f)
    assert g.counts["in"] + g.counts["out"] == 1


def test_layout_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "slots.json"
        save_layout(p, [("S1", [(0, 0), (10, 0), (10, 10)])], ((1, 2), (3, 4)), (640, 480))
        slots, gate_ = load_layout(p)
        assert slots[0].id == "S1" and gate_.p2 == (3.0, 4.0)


def test_end_to_end_pipeline_on_synthetic_video():
    """Scripted 'cars' on a synthetic lot -> real pipeline, writer, CSV, summary."""
    w, h, fps, n = 640, 360, 10, 200
    script = {}                               # frame -> detections
    for f in range(n):
        d = []
        if f < 160:                           # car A parks in S1 (x 100..200)
            x = min(150, 20 + f * 4) if f < 33 else 150
            d.append(car(x, 150, tid=1, w=70, h=50))
        if f < 40:                            # car B drives past along y=300
            d.append(car(20 + f * 15, 300, tid=2, w=70, h=50))
        script[f] = d
    # car C crosses gate line y=330 downward at x=400
    for f in range(100, 110):
        script[f] = script[f] + [car(400, 301 + (f - 100) * 6, tid=3, w=70, h=40)]
    with tempfile.TemporaryDirectory() as d:
        vid = str(Path(d) / "in.mp4")
        vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _ in range(n):
            vw.write(np.full((h, w, 3), 90, np.uint8))
        vw.release()
        state = {"i": -1}

        def fake_detector(frame):
            state["i"] += 1
            return script[state["i"]]

        cfg = {"occupancy": {"park_seconds": 2, "release_seconds": 3},
               "gate": {"in_direction": "positive"},
               "output": {"video": f"{d}/o.mp4", "events_csv": f"{d}/e.csv",
                          "timeline_csv": f"{d}/t.csv", "sqlite": f"{d}/p.db",
                          "summary": f"{d}/s.json"}}
        slots = [Slot("S1", [(80, 90), (220, 90), (220, 160), (80, 160)]),
                 Slot("S2", [(230, 90), (370, 90), (370, 160), (230, 160)])]
        summary = run(vid, fake_detector, slots, Gate((300, 330), (500, 330)),
                      cfg, display=False)
        assert summary["frames_processed"] == n
        assert summary["peak_occupied"] == 1 and summary["final_occupied"] == 0
        assert summary["gate"]["in"] == 1 and summary["gate"]["out"] == 0
        assert summary["slots"][0]["sessions"] == 1
        rows = Path(f"{d}/e.csv").read_text().strip().splitlines()
        assert len(rows) == 1 + 3, rows       # header + occupied + freed + vehicle_in
        out = cv2.VideoCapture(f"{d}/o.mp4")
        assert int(out.get(cv2.CAP_PROP_FRAME_COUNT)) == n
        out.release()
        assert json.loads(Path(f"{d}/s.json").read_text())["total_slots"] == 2
        import sqlite3
        con = sqlite3.connect(f"{d}/p.db")
        assert con.execute("select count(*) from events").fetchone()[0] == 3
        con.close()


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except Exception as ex:
                fails += 1
                import traceback
                print("FAIL", name, repr(ex))
                traceback.print_exc()
    sys.exit(1 if fails else 0)
