"""Offline SQLite storage with object, personal-object and room-memory tables."""
from __future__ import annotations

import queue
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

ALIASES = {
    "water bottle": "bottle",
    "drinking bottle": "bottle",
    "plastic bottle": "bottle",
    "cell phone": "cell phone",
    "mobile phone": "cell phone",
    "phone": "cell phone",
    "television": "tv",
    "telly": "tv",
    "couch": "couch",
    "sofa": "couch",
    "laptop computer": "laptop",
    "computer": "laptop",
    "back pack": "backpack",
    "clock": "watch",
    "wrist watch": "watch",
    "watch": "watch",
}


@dataclass(frozen=True)
class ObjectObservation:
    frame_id: int
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    observed_at: str
    room: str = "unknown"


class ObjectDatabase:
    def __init__(self, path: str | Path = "data/object_observations.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[ObjectObservation | None] = queue.Queue(maxsize=5000)
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._ready = threading.Event()
        self._thread.start()
        self._ready.wait()

    def record(self, observation: ObjectObservation) -> None:
        try:
            self._queue.put_nowait(observation)
        except queue.Full:
            pass

    def _canonical(self, label: str) -> str:
        clean = " ".join(label.lower().strip().split())
        return ALIASES.get(clean, clean)

    def last_seen(self, label: str):
        requested = self._canonical(label)
        with sqlite3.connect(self.path) as connection:
            personal = connection.execute(
                """
                SELECT name FROM personal_objects
                WHERE lower(name)=lower(?)
                   OR lower(name) LIKE ?
                   OR ? LIKE '%' || lower(name) || '%'
                ORDER BY length(name) DESC
                LIMIT 1
                """,
                (requested, f"%{requested.lower()}%", requested.lower()),
            ).fetchone()
            lookup = personal[0] if personal else requested
            row = connection.execute(
                """
                SELECT last_seen_at, confidence, x1, y1, x2, y2, room
                FROM objects
                WHERE label=? COLLATE NOCASE
                """,
                (lookup,),
            ).fetchone()
            if row is None and personal is None:
                candidates = connection.execute(
                    "SELECT label,last_seen_at,confidence,x1,y1,x2,y2,room FROM objects"
                ).fetchall()
                words = set(requested.split())
                for candidate in candidates:
                    stored_label = candidate[0].lower()
                    stored_words = set(stored_label.split())
                    if stored_label in requested or stored_words & words:
                        row = candidate[1:]
                        break
        if row is None:
            return None
        return (
            row[0],
            float(row[1]),
            tuple(float(value) for value in row[2:6]),
            row[6] or "unknown",
            lookup,
        )

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=3)

    @staticmethod
    def ensure_schema(path: str | Path) -> None:
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as connection:
            ObjectDatabase._create_schema(connection)
            connection.commit()

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS objects(
                label TEXT PRIMARY KEY COLLATE NOCASE,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                last_frame_id INTEGER NOT NULL,
                confidence REAL NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                room TEXT NOT NULL DEFAULT 'unknown'
            )
            """
        )
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(objects)").fetchall()
        }
        if "room" not in columns:
            connection.execute(
                "ALTER TABLE objects ADD COLUMN room TEXT NOT NULL DEFAULT 'unknown'"
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS object_sightings(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                observed_at TEXT NOT NULL,
                room TEXT NOT NULL DEFAULT 'unknown'
            )
            """
        )
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(object_sightings)").fetchall()
        }
        if "room" not in columns:
            connection.execute(
                "ALTER TABLE object_sightings ADD COLUMN room TEXT NOT NULL DEFAULT 'unknown'"
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sightings_label_time "
            "ON object_sightings(label, observed_at)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS personal_objects(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL COLLATE NOCASE,
                yolo_class TEXT NOT NULL COLLATE NOCASE,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS personal_object_crops(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                personal_object_id INTEGER NOT NULL,
                crop_path TEXT NOT NULL,
                hist_json TEXT NOT NULL,
                orb_descriptors BLOB,
                keypoint_count INTEGER NOT NULL,
                FOREIGN KEY(personal_object_id) REFERENCES personal_objects(id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rooms(
                name TEXT PRIMARY KEY COLLATE NOCASE,
                fingerprint_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    def _writer(self) -> None:
        with sqlite3.connect(self.path) as connection:
            self._create_schema(connection)
            old = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='object_observations'"
            ).fetchone()
            if old:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO objects(
                        label, first_seen_at, last_seen_at, last_frame_id,
                        confidence, x1, y1, x2, y2, room
                    )
                    SELECT o.label, MIN(o.observed_at), o.observed_at, o.frame_id,
                           o.confidence, o.x1, o.y1, o.x2, o.y2, 'unknown'
                    FROM object_observations o
                    JOIN (
                        SELECT label, MAX(observed_at) t
                        FROM object_observations
                        GROUP BY label
                    ) latest
                    ON o.label=latest.label AND o.observed_at=latest.t
                    GROUP BY o.label
                    """
                )
            connection.commit()
            self._ready.set()
            while True:
                item = self._queue.get()
                if item is None:
                    break
                connection.execute(
                    """
                    INSERT INTO objects VALUES(?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(label) DO UPDATE SET
                        last_seen_at=excluded.last_seen_at,
                        last_frame_id=excluded.last_frame_id,
                        confidence=excluded.confidence,
                        x1=excluded.x1,
                        y1=excluded.y1,
                        x2=excluded.x2,
                        y2=excluded.y2,
                        room=excluded.room
                    """,
                    (
                        item.label,
                        item.observed_at,
                        item.observed_at,
                        item.frame_id,
                        item.confidence,
                        item.x1,
                        item.y1,
                        item.x2,
                        item.y2,
                        item.room,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO object_sightings(
                        frame_id, label, confidence, x1, y1, x2, y2, observed_at, room
                    )
                    VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        item.frame_id,
                        item.label,
                        item.confidence,
                        item.x1,
                        item.y1,
                        item.x2,
                        item.y2,
                        item.observed_at,
                        item.room,
                    ),
                )
                connection.commit()
