# PUPIL / IRIS Smart Glasses

PUPIL is an offline Raspberry Pi 5 smart-glasses prototype for object-memory support during early cognitive decline. It uses a glasses-mounted webcam, YOLO11n object detection, a waist-mounted MPU6050 IMU, local SQLite memory, Vosk speech-to-text, Qwen2.5 0.5B fallback query logic, and Piper text-to-speech.

The system is designed to answer questions such as “PUPIL, where is my phone?” without sending indoor camera, voice, or memory data to the cloud.

## Repository structure

```text
.
├── main.py                     # Main runtime loop
├── imu.py                      # MPU6050 walking / turn / room-transition logic
├── nlp.py                      # Voice-command intent handling and memory answers
├── object_database.py          # Local SQLite object memory
├── stt.py                      # Vosk wake-word and speech recognition
├── tts.py                      # Piper speech output
├── vision_context.py           # Room-context inference from detected objects
├── shared_state.py             # Live shared system state
├── configs/                    # Runtime configuration files
├── docs/                       # Architecture, testing, safety and roadmap notes
├── hardware/3d-models/         # 3D-printable prototype enclosure files
├── models/                     # Model setup notes and small config files only
├── results/                    # Graphs and CSVs used in project evaluation
└── tools/                      # Testing and graph-generation scripts
```

## Hardware

- Raspberry Pi 5
- Logitech C270 webcam / microphone
- MPU6050 waist-mounted IMU
- MAX98357A I2S amplifier
- 4Ω speaker
- Power bank
- 3D-printed glasses / electronics enclosure

The STL files for the current prototype are in `hardware/3d-models/`.

## Runtime

On the Raspberry Pi:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-pi.txt
./run_pi.sh
```

## Model files

Large model binaries are intentionally not committed to GitHub. Place them locally when deploying:

- `models/yolo11n.pt`
- `models/en_US-lessac-medium.onnx`
- `models/vosk-model-small-en-us-0.15/`

Small model configuration files, setup notes, and reproducible testing code are included.

## Privacy and safety

PUPIL is a research prototype, not a medical device. It does not diagnose dementia, replace a caregiver, or guarantee exact object location. The main privacy design choice is that object detection, speech processing, memory lookup, and response generation run locally on the device.
