"""Offline SQLite storage with canonical object-label lookup."""
from __future__ import annotations
import queue, sqlite3, threading
from dataclasses import dataclass
from pathlib import Path

ALIASES = {
    "water bottle": "bottle", "drinking bottle": "bottle", "plastic bottle": "bottle",
    "cell phone": "cell phone", "mobile phone": "cell phone", "phone": "cell phone",
    "television": "tv", "telly": "tv", "couch": "couch", "sofa": "couch",
    "laptop computer": "laptop", "computer": "laptop", "back pack": "backpack",
    "clock": "watch", "wrist watch": "watch", "watch": "watch",
}

@dataclass(frozen=True)
class ObjectObservation:
    frame_id: int; label: str; confidence: float
    x1: float; y1: float; x2: float; y2: float; observed_at: str

class ObjectDatabase:
    def __init__(self, path: str | Path = "data/object_observations.sqlite3") -> None:
        self.path=Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[ObjectObservation|None]=queue.Queue(maxsize=5000)
        self._thread=threading.Thread(target=self._writer, daemon=True); self._ready=threading.Event()
        self._thread.start(); self._ready.wait()
    def record(self, observation: ObjectObservation) -> None:
        try: self._queue.put_nowait(observation)
        except queue.Full: pass
    def _canonical(self,label: str) -> str:
        clean=" ".join(label.lower().strip().split())
        return ALIASES.get(clean, clean)
    def last_seen(self,label: str):
        requested=self._canonical(label)
        with sqlite3.connect(self.path) as c:
            row=c.execute("SELECT last_seen_at,confidence,x1,y1,x2,y2 FROM objects WHERE label=? COLLATE NOCASE",(requested,)).fetchone()
            if row is None:
                # Match a descriptive phrase to a stored canonical COCO label.
                candidates=c.execute("SELECT label,last_seen_at,confidence,x1,y1,x2,y2 FROM objects").fetchall()
                words=set(requested.split())
                for candidate in candidates:
                    stored_label = candidate[0].lower()
                    stored_words=set(stored_label.split())
                    if stored_label == "clock" and requested == "watch":
                        row=candidate[1:]; break
                    if stored_label in requested or stored_words & words:
                        row=candidate[1:]; break
        return None if row is None else (row[0],float(row[1]),tuple(float(v) for v in row[2:6]))
    def close(self) -> None:
        self._queue.put(None); self._thread.join(timeout=3)
    def _writer(self) -> None:
        with sqlite3.connect(self.path) as c:
            c.execute("PRAGMA journal_mode=WAL"); c.execute("PRAGMA synchronous=NORMAL")
            c.execute("""CREATE TABLE IF NOT EXISTS objects(label TEXT PRIMARY KEY COLLATE NOCASE,first_seen_at TEXT NOT NULL,last_seen_at TEXT NOT NULL,last_frame_id INTEGER NOT NULL,confidence REAL NOT NULL,x1 REAL NOT NULL,y1 REAL NOT NULL,x2 REAL NOT NULL,y2 REAL NOT NULL)""")
            c.execute("""CREATE TABLE IF NOT EXISTS object_sightings(id INTEGER PRIMARY KEY AUTOINCREMENT,frame_id INTEGER NOT NULL,label TEXT NOT NULL,confidence REAL NOT NULL,x1 REAL NOT NULL,y1 REAL NOT NULL,x2 REAL NOT NULL,y2 REAL NOT NULL,observed_at TEXT NOT NULL)""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_sightings_label_time ON object_sightings(label,observed_at)")
            old=c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='object_observations'").fetchone()
            if old:
                c.execute("""INSERT OR IGNORE INTO objects(label,first_seen_at,last_seen_at,last_frame_id,confidence,x1,y1,x2,y2) SELECT o.label,MIN(o.observed_at),o.observed_at,o.frame_id,o.confidence,o.x1,o.y1,o.x2,o.y2 FROM object_observations o JOIN (SELECT label,MAX(observed_at) t FROM object_observations GROUP BY label) latest ON o.label=latest.label AND o.observed_at=latest.t GROUP BY o.label""")
            c.commit(); self._ready.set()
            while True:
                item=self._queue.get()
                if item is None: break
                c.execute("""INSERT INTO objects VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(label) DO UPDATE SET last_seen_at=excluded.last_seen_at,last_frame_id=excluded.last_frame_id,confidence=excluded.confidence,x1=excluded.x1,y1=excluded.y1,x2=excluded.x2,y2=excluded.y2""",(item.label,item.observed_at,item.observed_at,item.frame_id,item.confidence,item.x1,item.y1,item.x2,item.y2))
                c.execute("INSERT INTO object_sightings(frame_id,label,confidence,x1,y1,x2,y2,observed_at) VALUES(?,?,?,?,?,?,?,?)",(item.frame_id,item.label,item.confidence,item.x1,item.y1,item.x2,item.y2,item.observed_at)); c.commit()
