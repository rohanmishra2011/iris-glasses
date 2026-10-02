"""PUPIL: offline object memory, personal scanning, room context and voice."""
from __future__ import annotations

import argparse
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from config import load_tuning
from event_logger import EventLogger
from imu import MPU6050Worker
from nlp import answer
from object_database import ObjectDatabase, ObjectObservation
from personal_objects import PersonalObjectStore, crop_from_box
from room_memory import RoomMemory
from shared_state import SharedState
from stt import WakeWordListener
from tts import PiperSpeech
from vision_context import RoomClassifierWorker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="OpenCV USB camera index")
    parser.add_argument("--imgsz", type=int, default=480, help="YOLO inference image size")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--test-mode", action="store_true", help="Print live scores while logging CSVs")
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--no-room", action="store_true", help="Disable room recognition")
    parser.add_argument("--room-model", default="qwen2.5:0.5b-instruct")
    parser.add_argument("--room-interval", type=float, default=None)
    parser.add_argument("--tts", action="store_true", help="Play Piper responses when an audio output is connected")
    parser.add_argument(
        "--preview", action=argparse.BooleanOptionalAction, default=bool(os.environ.get("DISPLAY"))
    )
    return parser.parse_args()


def speak_or_print(speech: PiperSpeech | None, text: str) -> None:
    print(text, flush=True)
    if speech is not None:
        speech.speak(text)


def clean_phrase(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.strip(" ,.!?").split())


def main() -> None:
    args = parse_args()
    tuning = load_tuning(args.config)
    test_mode = args.test_mode or tuning.test_mode
    logger = EventLogger(test_mode=test_mode)
    state = SharedState()
    database = ObjectDatabase()
    room_memory = RoomMemory(database.path, tuning, logger)
    personal_objects = PersonalObjectStore(database.path, tuning, logger)
    imu_worker = MPU6050Worker(state, tuning=tuning, logger=logger)
    room_interval = args.room_interval if args.room_interval is not None else tuning.room_timer_seconds
    room_worker = None if args.no_room else RoomClassifierWorker(
        state,
        room_memory=room_memory,
        tuning=tuning,
        model=args.room_model,
        interval=room_interval,
    )

    latest_lock = threading.Lock()
    latest_frame: np.ndarray | None = None
    latest_detections: list[dict[str, Any]] = []

    def snapshot_latest() -> tuple[np.ndarray | None, list[dict[str, Any]]]:
        with latest_lock:
            return (
                None if latest_frame is None else latest_frame.copy(),
                [dict(detection) for detection in latest_detections],
            )

    def scan_object_command(listener: WakeWordListener | None, speech: PiperSpeech | None) -> str:
        frame, detections = snapshot_latest()
        target = (
            None
            if frame is None
            else personal_objects.detection_closest_to_centre(detections, frame.shape)
        )
        if frame is None or target is None:
            return "I do not see an object clearly enough to scan yet."
        yolo_class = str(target["yolo_class"])
        speak_or_print(speech, "What is this?")
        name = clean_phrase(listener.request_phrase(timeout=10.0) if listener is not None else "")
        if not name:
            return "I did not hear the object name."
        crops: list[np.ndarray] = []
        start = time.monotonic()
        interval = tuning.object_scan_seconds / max(1, tuning.object_scan_crop_count)
        while len(crops) < tuning.object_scan_crop_count and (
            time.monotonic() - start <= tuning.object_scan_seconds + 0.5
        ):
            frame_now, detections_now = snapshot_latest()
            if frame_now is not None:
                same_class = [
                    detection
                    for detection in detections_now
                    if detection.get("yolo_class") == yolo_class
                ]
                detection = personal_objects.detection_closest_to_centre(
                    same_class, frame_now.shape
                )
                if detection is not None:
                    crop = crop_from_box(frame_now, detection["box"])
                    if crop is not None:
                        crops.append(crop)
            time.sleep(interval)
        saved = personal_objects.save_personal_object(name, yolo_class, crops)
        if saved is None:
            return f"I could not save {name}."
        return f"I saved {name} as a personal {yolo_class}."

    def scan_room_command(listener: WakeWordListener | None, speech: PiperSpeech | None) -> str:
        speak_or_print(speech, "Please look around the room.")
        start = time.monotonic()
        time.sleep(tuning.room_scan_seconds)
        end = time.monotonic()
        speak_or_print(speech, "What room is this?")
        room_name = clean_phrase(listener.request_phrase(timeout=10.0) if listener is not None else "")
        if not room_name:
            return "I did not hear the room name."
        fingerprint = room_memory.scan_room(room_name, start, end)
        if not fingerprint:
            return f"I could not save {room_name}. I did not see enough room landmarks."
        return f"I saved this room as {room_name}."

    imu_worker.start()
    if room_worker is not None:
        room_worker.start()

    state.set_status(yolo="loading")
    model = YOLO("yolo11n.pt")
    state.set_status(yolo="running")

    listener = None
    speech = PiperSpeech() if args.tts else None
    if speech is not None:
        speak_or_print(speech, "Calibrating, please stay still")

    if not args.no_voice:
        def handle_command(command: str, speech_end_at: float | None = None) -> None:
            state.set_status(
                voice_status="processing",
                last_command=command,
                last_response="",
                last_command_at=datetime.now(timezone.utc).isoformat(),
            )
            clean = command.lower().strip()
            try:
                if "scan this object" in clean or "scan object" in clean:
                    response = scan_object_command(listener, speech)
                elif "scan this room" in clean or "scan room" in clean:
                    response = scan_room_command(listener, speech)
                else:
                    response = answer(command, database, state.snapshot())
            except Exception as exc:
                response = "I could not process that command offline."
                state.set_status(voice_status="error")
                print(f"[voice command] {exc}", flush=True)
            else:
                state.set_status(voice_status="listening")
            answer_start = time.monotonic()
            latency = None if speech_end_at is None else max(0.0, answer_start - speech_end_at)
            logger.log(
                "queries",
                question=command,
                answer=response,
                latency_seconds="" if latency is None else latency,
            )
            state.set_status(last_response=response)
            speak_or_print(speech, response)

        try:
            listener = WakeWordListener("models/vosk-model-small-en-us-0.15")
            listener.start(handle_command)
            state.set_status(voice_status="listening")
        except Exception as exc:
            state.set_status(voice_status="error", last_response="Microphone unavailable.")
            print(f"[voice startup] {exc}", flush=True)
    else:
        state.set_status(voice_status="disabled")

    camera = None
    next_camera_retry = 0.0
    frame_id = 0
    last_imu_event_id = 0
    fps_start = time.monotonic()
    fps_frames = 0
    try:
        while True:
            if camera is None:
                if time.monotonic() >= next_camera_retry:
                    state.set_status(camera="opening")
                    candidate = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
                    candidate.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    candidate.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    candidate.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    candidate.set(cv2.CAP_PROP_FPS, 30)
                    if candidate.isOpened():
                        camera = candidate
                        state.set_status(camera="running")
                    else:
                        candidate.release()
                        state.set_status(camera="disconnected")
                        next_camera_retry = time.monotonic() + 2.0
                if camera is None:
                    state.publish(None, [], frame_id=frame_id, camera="disconnected")
                    time.sleep(0.25)
                    continue
            ok, frame = camera.read()
            if not ok:
                camera.release()
                camera = None
                state.set_status(camera="read_error")
                next_camera_retry = time.monotonic() + 2.0
                continue
            if float(frame.mean()) < 5.0:
                state.set_status(camera="warming_up")
                time.sleep(0.05)
                continue

            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            result = model.predict(frame, conf=0.40, imgsz=args.imgsz, verbose=False)[0]
            observed_at = datetime.now(timezone.utc).isoformat()
            snapshot = state.snapshot()
            current_room = str(snapshot.get("room") or "unknown")
            detections = []
            object_counts: dict[str, int] = {}
            for box, confidence, class_id in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist(),
                strict=True,
            ):
                raw_label = result.names[int(class_id)]
                yolo_label = "watch" if raw_label == "clock" else raw_label
                if raw_label == "clock":
                    result.names[int(class_id)] = "watch"
                object_counts[yolo_label] = object_counts.get(yolo_label, 0) + 1
                crop = crop_from_box(frame, box)
                personal_name = None
                if crop is not None:
                    personal_name, _scores = personal_objects.match(yolo_label, crop)
                stored_label = personal_name or yolo_label
                database.record(
                    ObjectObservation(
                        frame_id,
                        stored_label,
                        float(confidence),
                        *map(float, box),
                        observed_at,
                        current_room,
                    )
                )
                detections.append(
                    {
                        "label": stored_label,
                        "yolo_class": yolo_label,
                        "confidence": float(confidence),
                        "box": [float(value) for value in box],
                    }
                )
            now = time.monotonic()
            room_memory.observe(object_counts, now)
            with latest_lock:
                latest_frame = frame.copy()
                latest_detections = [dict(detection) for detection in detections]

            snapshot = state.snapshot()
            imu_event_id = int(snapshot.get("imu_turn_event_id") or 0)
            force_room_update = (
                imu_event_id != last_imu_event_id
                and bool(snapshot.get("imu_room_transition_likely"))
            )
            if force_room_update:
                last_imu_event_id = imu_event_id
                print("[imu] transition trigger; recognising room after 5 s", flush=True)
            if room_worker is not None:
                room_worker.submit_objects(
                    object_counts,
                    force=force_room_update,
                    trigger_source="IMU" if force_room_update else "timer",
                )

            display = result.plot()
            room = snapshot["room"]
            cv2.putText(
                display,
                f"Room: {room}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 180, 0),
                2,
            )
            state.publish(
                display,
                detections,
                frame_id=frame_id,
                camera="running",
                yolo="running",
            )
            if args.preview:
                cv2.imshow("PUPIL - YOLO11n", display)
            frame_id += 1
            fps_frames += 1
            if test_mode and time.monotonic() - fps_start >= 5.0:
                fps = fps_frames / (time.monotonic() - fps_start)
                print(f"[fps] {fps:.2f}", flush=True)
                fps_start = time.monotonic()
                fps_frames = 0
            if args.preview and cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if listener is not None:
            listener.stop()
        imu_worker.stop()
        if room_worker is not None:
            room_worker.stop()
        database.close()
        if camera is not None:
            camera.release()
        if args.preview:
            cv2.destroyAllWindows()
        state.set_status(camera="stopped")


if __name__ == "__main__":
    main()
