# VSGlasses

VSGlasses is a Raspberry Pi–compatible smart-glasses project for contextual memory support for people living with Alzheimer’s disease and other forms of dementia. It is not intended to replace a caregiver, diagnose a condition, or provide blind-navigation assistance.

## Current state

The project has intentionally been reset to one small, testable baseline:

- OpenCV camera capture and live preview;
- YOLO11n live object detection;
- `q` exits the preview;
- no speech, SQL database, face recognition, room mapping or Ollama runtime yet.

The only active runtime entry point is `main.py`:

```bash
source .venv/bin/activate
python main.py
```

Install the detector dependency if needed:

```bash
python -m pip install ultralytics opencv-python
```

The first run may download `yolo11n.pt`. A desktop run is not Raspberry Pi validation: measure FPS, latency, memory, heat and power on the target Pi.

## Planned capabilities

1. Conservative live facial recognition with consent-based enrollment.
2. A local memory store for people, objects, observations, timestamps and confidence.
3. “Where was this object last seen?” retrieval based on confirmed observations.
4. Autonomous house/room mapping using camera motion, an IMU and future SLAM.
5. Object positions anchored to the map and updated over time.
6. Voice input and output, added only after the visual baseline is stable.

The future system should save structured events rather than every frame. It must distinguish “last observed” from “currently located,” expose uncertainty, and ask for confirmation before saving important memories or enrolling a person. Faces, voices, locations and memory records are sensitive data and should remain local by default.

## Rules for future agents

- Raspberry Pi compatibility is a primary constraint: prefer small offline models, low RAM use and Raspberry Pi OS-compatible dependencies.
- One model must have one clearly defined job. Do not add Ollama or a second detector to the live loop without measured justification.
- Keep OpenCV GUI work on the main thread. Move blocking inference, audio or database work off that thread when later modules require it.
- Do not claim medical benefit, reliable identity or exact object location without testing and confidence handling.
- Add one module at a time and verify it on target hardware.
- Report hardware tests separately from syntax checks or reasoning.

## Future architecture

```text
Camera + IMU
    -> fast visual perception
    -> structured observations with timestamps/confidence
    -> local memory and map
    -> consent-aware assistance layer
    -> optional voice output
```

The next milestone is proving that YOLO11n and OpenCV run reliably on the Raspberry Pi with acceptable latency and power use. Only then should facial recognition and the memory/map system be added.
