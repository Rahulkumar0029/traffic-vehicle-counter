"""Click-to-define parking slots and the entry/exit gate line.

python define_slots.py --source parking.mp4 [--frame 0] [--out slots.json]

Mouse : left-click adds a corner point.
Keys  : ENTER / n  finish current slot (polygon, 3+ points)
        r          toggle RECTANGLE mode (2 clicks = opposite corners)
        g          toggle GATE mode (2 clicks = gate line, optional)
        u          undo last point / last slot
        s          save to slots.json
        q / Esc    quit
Draw slots on a frame where cars are NOT covering the slot markings if you can.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from parking.slots import save_layout

COLORS = {"slot": (60, 200, 60), "cur": (0, 220, 255), "gate": (255, 160, 0)}


class Editor:
    def __init__(self):
        self.slots = []        # list of (id, [(x, y), ...])
        self.points = []       # points of the slot being drawn
        self.gate = None       # ((x, y), (x, y))
        self.gate_points = []
        self.rect_mode = False
        self.gate_mode = False

    def click(self, x, y):
        if self.gate_mode:
            self.gate_points.append((x, y))
            if len(self.gate_points) == 2:
                self.gate = (self.gate_points[0], self.gate_points[1])
                self.gate_points = []
                self.gate_mode = False
            return
        self.points.append((x, y))
        if self.rect_mode and len(self.points) == 2:
            (x1, y1), (x2, y2) = self.points
            self.points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            self.finish()

    def finish(self):
        if len(self.points) >= 3:
            self.slots.append((f"S{len(self.slots) + 1}", list(self.points)))
        self.points = []

    def undo(self):
        if self.gate_mode and self.gate_points:
            self.gate_points.pop()
        elif self.points:
            self.points.pop()
        elif self.slots:
            self.slots.pop()

    def render(self, base):
        img = base.copy()
        for sid, poly in self.slots:
            pts = np.array(poly, np.int32)
            cv2.polylines(img, [pts], True, COLORS["slot"], 2)
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            cv2.putText(img, sid, (int(cx) - 12, int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
        for p in self.points:
            cv2.circle(img, p, 4, COLORS["cur"], -1)
        if len(self.points) > 1:
            cv2.polylines(img, [np.array(self.points, np.int32)], False,
                          COLORS["cur"], 1)
        if self.gate:
            cv2.line(img, self.gate[0], self.gate[1], COLORS["gate"], 3)
        for p in self.gate_points:
            cv2.circle(img, p, 5, COLORS["gate"], -1)
        mode = "GATE" if self.gate_mode else ("RECT" if self.rect_mode else "POLY")
        msg = f"mode:{mode}  slots:{len(self.slots)}  [Enter]=finish [r]=rect [g]=gate [u]=undo [s]=save [q]=quit"
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(img, msg, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img


def grab_frame(source, index):
    src = str(source)
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {src}")
    if index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read a frame")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default="slots.json")
    args = ap.parse_args()

    frame = grab_frame(args.source, args.frame)
    h, w = frame.shape[:2]
    ed = Editor()
    win = "Define slots"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lambda e, x, y, f, p: ed.click(x, y)
                         if e == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        cv2.imshow(win, ed.render(frame))
        k = cv2.waitKey(30) & 0xFF
        if k in (13, ord("n")):
            ed.finish()
        elif k == ord("r"):
            ed.rect_mode = not ed.rect_mode
            ed.gate_mode = False
        elif k == ord("g"):
            ed.gate_mode = not ed.gate_mode
        elif k == ord("u"):
            ed.undo()
        elif k == ord("s"):
            save_layout(args.out, ed.slots, ed.gate, (w, h))
            print(f"Saved {len(ed.slots)} slots"
                  f"{' + gate' if ed.gate else ''} to {Path(args.out).resolve()}")
        elif k in (27, ord("q")):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
