# Model files

Large model binaries are intentionally not committed to GitHub.

Expected local deployment files:

- `yolo11n.pt` for YOLO11n object detection
- `en_US-lessac-medium.onnx` for Piper speech output
- `vosk-model-small-en-us-0.15/` for offline speech recognition

The small Piper JSON config is committed because it is lightweight and useful for deployment. Downloaded model weights should stay local on the Raspberry Pi or development machine.
