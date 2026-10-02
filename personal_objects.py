"""Personal object enrolment and matching using colour and ORB features."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

import cv2
import numpy as np

from config import Tuning
from event_logger import EventLogger
from object_database import ObjectDatabase


def crop_from_box(frame: np.ndarray, box: list[float] | tuple[float, ...]) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


class PersonalObjectStore:
    def __init__(
        self,
        db_path: str | Path,
        tuning: Tuning,
        logger: EventLogger,
        crop_dir: str | Path = "data/personal_object_crops",
    ) -> None:
        self.db_path = Path(db_path)
        self.tuning = tuning
        self.logger = logger
        self.crop_dir = Path(crop_dir)
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        self.orb = cv2.ORB_create(nfeatures=tuning.orb_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._last_compare_by_class: dict[str, float] = {}
        self._profiles_loaded_at = 0.0
        self._profiles: dict[str, list[dict[str, Any]]] = {}
        ObjectDatabase.ensure_schema(self.db_path)

    @staticmethod
    def detection_closest_to_centre(
        detections: list[dict[str, Any]], frame_shape: tuple[int, ...]
    ) -> dict[str, Any] | None:
        if not detections:
            return None
        height, width = frame_shape[:2]
        cx, cy = width / 2.0, height / 2.0
        return min(
            detections,
            key=lambda detection: (
                ((detection["box"][0] + detection["box"][2]) / 2.0 - cx) ** 2
                + ((detection["box"][1] + detection["box"][3]) / 2.0 - cy) ** 2
            ),
        )

    def save_personal_object(
        self, name: str, yolo_class: str, crops: list[np.ndarray]
    ) -> int | None:
        clean_name = " ".join(name.strip().split())
        clean_class = " ".join(yolo_class.strip().lower().split())
        if not clean_name or not clean_class or not crops:
            return None
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as connection:
            ObjectDatabase._create_schema(connection)
            connection.execute(
                """
                INSERT INTO personal_objects(name, yolo_class, created_at)
                VALUES(?,?,?)
                ON CONFLICT(name) DO UPDATE SET
                    yolo_class=excluded.yolo_class,
                    created_at=excluded.created_at
                """,
                (clean_name, clean_class, now),
            )
            object_id = int(
                connection.execute(
                    "SELECT id FROM personal_objects WHERE name=? COLLATE NOCASE",
                    (clean_name,),
                ).fetchone()[0]
            )
            connection.execute(
                "DELETE FROM personal_object_crops WHERE personal_object_id=?",
                (object_id,),
            )
            for index, crop in enumerate(crops):
                resized = cv2.resize(
                    crop,
                    (self.tuning.personal_crop_size, self.tuning.personal_crop_size),
                    interpolation=cv2.INTER_AREA,
                )
                path = self.crop_dir / f"{object_id}_{index}.png"
                cv2.imwrite(str(path), resized)
                hist = self._hsv_hist(resized)
                descriptors, keypoint_count = self._orb_descriptors(resized)
                connection.execute(
                    """
                    INSERT INTO personal_object_crops(
                        personal_object_id, crop_path, hist_json,
                        orb_descriptors, keypoint_count
                    )
                    VALUES(?,?,?,?,?)
                    """,
                    (
                        object_id,
                        str(path),
                        json.dumps(hist.tolist()),
                        None if descriptors is None else descriptors.tobytes(),
                        keypoint_count,
                    ),
                )
            connection.commit()
        self._profiles_loaded_at = 0.0
        return object_id

    def match(self, yolo_class: str, crop: np.ndarray) -> tuple[str | None, dict[str, float]]:
        clean_class = " ".join(yolo_class.strip().lower().split())
        now = time.monotonic()
        if now - self._last_compare_by_class.get(clean_class, 0.0) < (
            self.tuning.object_match_min_interval_seconds
        ):
            return None, {"colour": 0.0, "orb": 0.0, "combined": 0.0}
        self._last_compare_by_class[clean_class] = now
        profiles = self._profiles_for_class(clean_class)
        if not profiles:
            return None, {"colour": 0.0, "orb": 0.0, "combined": 0.0}

        resized = cv2.resize(
            crop,
            (self.tuning.personal_crop_size, self.tuning.personal_crop_size),
            interpolation=cv2.INTER_AREA,
        )
        query_hist = self._hsv_hist(resized)
        query_descriptors, query_keypoints = self._orb_descriptors(resized)

        best_name: str | None = None
        best_scores = {"colour": 0.0, "orb": 0.0, "combined": 0.0}
        for profile in profiles:
            colour = float(cv2.compareHist(query_hist, profile["hist"], cv2.HISTCMP_CORREL))
            colour = max(0.0, min(1.0, colour))
            use_colour_only = (
                query_keypoints < self.tuning.orb_min_keypoints
                or profile["keypoint_count"] < self.tuning.orb_min_keypoints
                or query_descriptors is None
                or profile["descriptors"] is None
            )
            if use_colour_only:
                orb_score = 0.0
                combined = colour
            else:
                matches = self.matcher.knnMatch(
                    query_descriptors, profile["descriptors"], k=2
                )
                good = []
                for pair in matches:
                    if len(pair) < 2:
                        continue
                    first, second = pair
                    if first.distance < self.tuning.orb_ratio * second.distance:
                        good.append(first)
                denominator = max(1, min(query_keypoints, profile["keypoint_count"]))
                orb_score = min(1.0, len(good) / denominator)
                combined = 0.5 * colour + 0.5 * orb_score
            if combined > best_scores["combined"]:
                best_name = profile["name"]
                best_scores = {
                    "colour": colour,
                    "orb": orb_score,
                    "combined": combined,
                }

        decision = "personal" if best_scores["combined"] >= self.tuning.match_threshold else "generic"
        self.logger.log(
            "object_matches",
            yolo_class=clean_class,
            personal_object=best_name or "",
            colour_score=best_scores["colour"],
            orb_score=best_scores["orb"],
            combined_score=best_scores["combined"],
            decision=decision,
        )
        if decision == "personal":
            return best_name, best_scores
        return None, best_scores

    def _profiles_for_class(self, yolo_class: str) -> list[dict[str, Any]]:
        if time.monotonic() - self._profiles_loaded_at > 2.0:
            self._load_profiles()
        return self._profiles.get(yolo_class, [])

    def _load_profiles(self) -> None:
        profiles: dict[str, list[dict[str, Any]]] = {}
        with sqlite3.connect(self.db_path) as connection:
            ObjectDatabase._create_schema(connection)
            rows = connection.execute(
                """
                SELECT p.name, p.yolo_class, c.hist_json, c.orb_descriptors, c.keypoint_count
                FROM personal_objects p
                JOIN personal_object_crops c ON c.personal_object_id=p.id
                """
            ).fetchall()
        for name, yolo_class, hist_json, descriptors_blob, keypoint_count in rows:
            descriptors = None
            if descriptors_blob:
                descriptors = np.frombuffer(descriptors_blob, dtype=np.uint8).reshape(-1, 32)
            profiles.setdefault(yolo_class.lower(), []).append(
                {
                    "name": name,
                    "hist": np.array(json.loads(hist_json), dtype=np.float32),
                    "descriptors": descriptors,
                    "keypoint_count": int(keypoint_count),
                }
            )
        self._profiles = profiles
        self._profiles_loaded_at = time.monotonic()

    def _hsv_hist(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            [self.tuning.hsv_h_bins, self.tuning.hsv_s_bins],
            [0, 180, 0, 256],
        )
        return cv2.normalize(hist, hist).astype(np.float32)

    def _orb_descriptors(self, image: np.ndarray) -> tuple[np.ndarray | None, int]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return descriptors, len(keypoints)
