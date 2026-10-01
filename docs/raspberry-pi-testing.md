# Raspberry Pi testing

Target: Raspberry Pi 5, 4 GB, 64-bit Raspberry Pi OS, Python 3.11, USB webcam.

## Transfer

Do not copy `.venv`, caches, generated output, or the local development
database. Virtual environments and compiled packages are operating-system and
CPU specific.

Transfer the source tree while excluding:

```text
.venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
outputs/
data/
```

The ONNX file under `models/` must be included.

## Raspberry Pi preparation

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev v4l-utils ffmpeg libgl1 libglib2.0-0

cd VSGlasses
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

Confirm the webcam:

```bash
v4l2-ctl --list-devices
VSGLASSES_RUN_HARDWARE_TESTS=1 pytest tests/hardware/test_usb_camera.py -v
```

## First interactive run

Run from the Raspberry Pi desktop:

```bash
python main.py --config configs/raspberry_pi_development.toml
```

Verify:

- the preview opens;
- Diagnostics toggles;
- object boxes appear;
- track recovery works after short occlusion;
- Ask returns a last-seen answer;
- clean shutdown prints one session summary;
- `data/object_memory.sqlite3` and `outputs/perception.jsonl` are created.

## Initial measurements

Record camera FPS, detection FPS, current and average inference latency, frame
latency, CPU, RAM, and temperature. Do not change the model until these baseline
numbers are captured.

Temperature:

```bash
vcgencmd measure_temp
```

After interactive validation, run the camera-only soak procedure in
`phase-1-validation.md`. A full perception soak command should be added after
the first Pi measurements establish sustainable inference settings.
