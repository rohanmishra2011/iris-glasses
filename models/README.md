# Model files

The repository stores code and lightweight configuration only. Downloaded model
binaries are intentionally excluded from GitHub.

Expected local model files on the Raspberry Pi:

- `yolo11n.pt` in the project root for YOLO object detection.
- `models/vosk-model-small-en-us-0.15/` for offline speech-to-text.
- `models/en_US-lessac-medium.onnx` for Piper text-to-speech.
- `models/en_US-lessac-medium.onnx.json` for the Piper voice configuration.

The Pi currently has these files installed locally; they are not committed
because they are large third-party binaries.
