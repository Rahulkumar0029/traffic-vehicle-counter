"""Writes events to CSV and (optionally) SQLite."""
import csv
import sqlite3
from pathlib import Path

COLUMNS = ["time_s", "frame", "event", "slot", "track_id", "label", "duration_s"]


class EventLog:
    def __init__(self, csv_path=None, sqlite_path=None):
        self._csv_file = self._csv = self._db = None
        if csv_path:
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(csv_path, "w", newline="")
            self._csv = csv.DictWriter(self._csv_file, fieldnames=COLUMNS)
            self._csv.writeheader()
        if sqlite_path:
            Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(sqlite_path)
            self._db.execute("DROP TABLE IF EXISTS events")
            self._db.execute(
                "CREATE TABLE events (time_s REAL, frame INTEGER, event TEXT, "
                "slot TEXT, track_id INTEGER, label TEXT, duration_s REAL)")

    def write(self, ev: dict):
        if self._csv:
            self._csv.writerow({k: ev.get(k, "") for k in COLUMNS})
            self._csv_file.flush()
        if self._db:
            self._db.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                tuple(None if ev.get(k, "") == "" else ev.get(k) for k in COLUMNS))
            self._db.commit()

    def close(self):
        if self._csv_file:
            self._csv_file.close()
        if self._db:
            self._db.close()
