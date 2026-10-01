"""VSGlasses: one USB camera, one YOLO model, voice, memory and dashboard."""
import argparse
import os
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from ultralytics import YOLO

from object_database import ObjectDatabase, ObjectObservation
from stt import WakeWordListener
from tts import PiperSpeech
from nlp import answer
from dashboard.server import DashboardServer
from shared_state import SharedState
from system_health import SystemHealthWorker
from imu import MPU6050Worker
from vision_context import RoomClassifierWorker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="OpenCV USB camera index")
    parser.add_argument("--imgsz", type=int, default=480, help="YOLO inference image size")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--no-room", action="store_true", help="Disable Qwen room classification")
    parser.add_argument("--room-model", default="qwen2.5:0.5b-instruct")
    parser.add_argument("--room-interval", type=float, default=30.0)
    parser.add_argument("--tts", action="store_true", help="Play Piper responses when an audio output is connected")
    parser.add_argument(
        "--preview", action=argparse.BooleanOptionalAction, default=bool(os.environ.get("DISPLAY"))
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = SharedState(jpeg_quality=70, stream_fps=8)
    dashboard = None if args.no_dashboard else DashboardServer(state, args.host, args.port)
    if dashboard is not None:
        dashboard.start()
    health_worker = SystemHealthWorker(state)
    imu_worker = MPU6050Worker(state)
    room_worker = None if args.no_room else RoomClassifierWorker(
        state, model=args.room_model, interval=args.room_interval
    )
    health_worker.start()
    imu_worker.start()
    if room_worker is not None:
        room_worker.start()

    state.set_status(yolo="loading")
    model = YOLO("yolo11n.pt")
    state.set_status(yolo="running")
    database = ObjectDatabase()
    listener = None
    speech = PiperSpeech() if args.tts else None
    if not args.no_voice:
        def handle_command(command: str) -> None:
            state.set_status(
                voice_status="processing",
                last_command=command,
                last_response="",
                last_command_at=datetime.now(timezone.utc).isoformat(),
            )
            try:
                response = answer(command, database, state.snapshot())
            except Exception as exc:
                response = "I could not process that command offline."
                state.set_status(voice_status="error")
                print(f"[voice command] {exc}", flush=True)
            else:
                state.set_status(voice_status="listening")
            state.set_status(last_response=response)
            if speech is not None:
                speech.speak(response)

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
    last_room_turn_event_id = 0
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
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        placeholder, "USB camera disconnected", (105, 245),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2,
                    )
                    state.publish(placeholder, [], frame_id=frame_id, camera="disconnected")
                    time.sleep(0.25)
                    continue
            ok, frame = camera.read()
            if not ok:
                camera.release()
                camera = None
                state.set_status(camera="read_error")
                next_camera_retry = time.monotonic() + 2.0
                continue
            # USB webcams can return several all-black startup frames. Never
            # run detection or send those frames to the vision model.
            if float(frame.mean()) < 5.0:
                state.set_status(camera="warming_up")
                time.sleep(0.05)
                continue
            # The camera is mounted sideways on the glasses. Rotate the
            # shared frame so YOLO, room inference, and the dashboard agree.
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            result = model.predict(frame, conf=0.40, imgsz=args.imgsz, verbose=False)[0]
            observed_at = datetime.now(timezone.utc).isoformat()
            detections = []
            object_counts = {}
            for box, confidence, class_id in zip(
                result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                result.boxes.cls.tolist(), strict=True
            ):
                raw_label = result.names[int(class_id)]
                label = "watch" if raw_label == "clock" else raw_label
                if raw_label == "clock":
                    result.names[int(class_id)] = "watch"
                object_counts[label] = object_counts.get(label, 0) + 1
                database.record(ObjectObservation(
                    frame_id, label, float(confidence),
                    *map(float, box), observed_at
                ))
                detections.append({
                    "label": label,
                    "confidence": float(confidence),
                    "box": [float(value) for value in box],
                })
            snapshot = state.snapshot()
            turn_event_id = int(snapshot.get("imu_turn_event_id") or 0)
            force_room_update = (
                turn_event_id != last_room_turn_event_id
                and bool(snapshot.get("imu_room_transition_likely"))
            )
            if force_room_update:
                last_room_turn_event_id = turn_event_id
                print(
                    "[imu] room transition likely after "
                    f"{snapshot.get('imu_turn_direction') or 'unknown'} turn; "
                    "refreshing room label",
                    flush=True,
                )
            if room_worker is not None:
                room_worker.submit_objects(object_counts, force=force_room_update)
            display = result.plot()
            room = snapshot["room"]
            cv2.putText(
                display, f"Room: {room}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 180, 0), 2,
            )
            state.publish(
                display,
                detections,
                frame_id=frame_id,
                camera="running",
                yolo="running",
            )
            if args.preview:
                cv2.imshow("Smart Glasses - YOLO11n", display)
            frame_id += 1
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
        health_worker.stop()
        database.close()
        if camera is not None:
            camera.release()
        if args.preview:
            cv2.destroyAllWindows()
        state.set_status(camera="stopped")
        if dashboard is not None:
            dashboard.stop()


if __name__ == "__main__":
    main()
