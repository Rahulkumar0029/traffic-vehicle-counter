"""Main processing loop: video -> detect/track -> occupancy + gate -> output."""
import csv
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .eventlog import EventLog
from .gate import GateCounter
from .occupancy import OccupancyEngine

GREEN, RED, WHITE, YELLOW = (60, 200, 60), (50, 50, 230), (255, 255, 255), (0, 220, 255)


def open_source(source):
    """Accepts a file path, webcam index ('0'), or a stream URL (rtsp/http)."""
    src = str(source)
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        if "://" not in src and not Path(src).is_file():
            raise FileNotFoundError(f"Video not found: {src}")
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {src}")
    return cap


def _fmt(seconds):
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def _draw_slots(frame, snap):
    overlay = frame.copy()
    for s in snap:
        cv2.fillPoly(overlay, [s["polygon"]], RED if s["occupied"] else GREEN)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    for s in snap:
        color = RED if s["occupied"] else GREEN
        cv2.polylines(frame, [s["polygon"]], True, color, 2)
        text = s["id"] if s["parked_s"] is None else f'{s["id"]} {_fmt(s["parked_s"])}'
        cv2.putText(frame, text, (int(s["cx"]) - 28, int(s["cy"])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2, cv2.LINE_AA)


def _draw_hud(frame, engine, gate, proc_fps):
    lines = [f"Slots: {engine.total()}   Occupied: {engine.occupied_count()}   "
             f"Free: {engine.free_count()}"]
    if gate:
        lines.append(f"Gate  IN: {gate.counts['in']}   OUT: {gate.counts['out']}")
    lines.append(f"Processing: {proc_fps:.1f} FPS")
    box_w, box_h = 430, 14 + 28 * len(lines)
    roi = frame[8:8 + box_h, 8:8 + box_w]
    frame[8:8 + box_h, 8:8 + box_w] = (roi * 0.35).astype(np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (16, 34 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, WHITE, 2, cv2.LINE_AA)
    if engine.free_count() == 0:
        cv2.putText(frame, "LOT FULL", (16, 34 + 28 * len(lines) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 3, cv2.LINE_AA)


def run(source, detector, slots, gate_def, cfg, display=True, max_frames=0):
    """detector: callable(frame) -> list[Detection]. Returns summary dict."""
    cap = open_source(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    occ = cfg.get("occupancy", {})
    gcfg = cfg.get("gate", {})
    anchor = cfg.get("anchor", "bottom")
    engine = OccupancyEngine(
        slots, fps, width,
        park_seconds=occ.get("park_seconds", 2.0),
        release_seconds=occ.get("release_seconds", 3.0),
        min_presence=occ.get("min_presence", 0.8),
        move_threshold_frac=occ.get("move_threshold_frac", 0.02),
        anchor=anchor)
    gate = (GateCounter(gate_def, fps, gcfg.get("cooldown_seconds", 2.0),
                        gcfg.get("in_direction", "positive"), anchor)
            if gate_def else None)

    out = cfg.get("output", {})
    log = EventLog(out.get("events_csv"), out.get("sqlite"))
    writer = None
    if out.get("video"):
        Path(out["video"]).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(out["video"], cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
    timeline_f = timeline = None
    if out.get("timeline_csv"):
        Path(out["timeline_csv"]).parent.mkdir(parents=True, exist_ok=True)
        timeline_f = open(out["timeline_csv"], "w", newline="")
        timeline = csv.writer(timeline_f)
        timeline.writerow(["time_s", "occupied", "free"])

    trails = {}
    class_seen = {}
    n_events = 0
    frame_idx = 0
    t0 = time.perf_counter()
    show = display
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = detector(frame)

            events = engine.update(dets, frame_idx)
            if gate:
                events += gate.update(dets, frame_idx)
            for ev in events:
                log.write(ev)
                n_events += 1
                print(f"[{ev['time_s']:7.2f}s] {ev['event']:14s} "
                      f"{ev['slot'] or ''} {ev['label']} {ev['duration_s']}")

            if timeline and frame_idx % max(1, round(fps)) == 0:
                timeline.writerow([round(frame_idx / fps, 1),
                                   engine.occupied_count(), engine.free_count()])

            # ---- draw
            _draw_slots(frame, engine.snapshot(frame_idx))
            if gate:
                a, b = gate.gate.p1, gate.gate.p2
                cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                         YELLOW, 3)
                s, e = gate.in_arrow()
                cv2.arrowedLine(frame, s, e, YELLOW, 3, tipLength=0.35)
                cv2.putText(frame, "IN", (e[0] + 6, e[1]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, YELLOW, 2, cv2.LINE_AA)
            for d in dets:
                x1, y1, x2, y2 = (int(v) for v in d.bbox)
                class_seen.setdefault(d.track_id, d.label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 2)
                cv2.putText(frame, f"{d.label} #{d.track_id}", (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2, cv2.LINE_AA)
                p = tuple(int(v) for v in d.anchor(anchor))
                cv2.circle(frame, p, 4, (0, 0, 255), -1)
                tr = trails.setdefault(d.track_id, deque(maxlen=25))
                tr.append(p)
                for i in range(1, len(tr)):
                    cv2.line(frame, tr[i - 1], tr[i], YELLOW, 1)
            if frame_idx % 300 == 0:
                live = {d.track_id for d in dets}
                for t in [t for t in trails if t not in live]:
                    trails.pop(t, None)

            elapsed = max(1e-6, time.perf_counter() - t0)
            _draw_hud(frame, engine, gate, (frame_idx + 1) / elapsed)

            if writer:
                writer.write(frame)
            if show:
                try:
                    cv2.imshow("Smart Parking", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                except cv2.error:
                    show = False  # headless OpenCV build
            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        log.close()
        if timeline_f:
            timeline_f.close()

    elapsed = time.perf_counter() - t0
    summary = {
        "frames_processed": frame_idx,
        "video_duration_s": round(frame_idx / fps, 1),
        "processing_fps": round(frame_idx / max(elapsed, 1e-6), 1),
        "total_slots": engine.total(),
        "final_occupied": engine.occupied_count(),
        "final_free": engine.free_count(),
        "peak_occupied": engine.peak_occupied,
        "slots": engine.slot_stats(frame_idx),
        "gate": ({"in": gate.counts["in"], "out": gate.counts["out"],
                  "in_by_class": gate.by_class["in"],
                  "out_by_class": gate.by_class["out"]} if gate else None),
        "unique_vehicles_tracked": len(class_seen),
        "events_logged": n_events,
    }
    if out.get("summary"):
        Path(out["summary"]).parent.mkdir(parents=True, exist_ok=True)
        Path(out["summary"]).write_text(json.dumps(summary, indent=2))
    return summary
