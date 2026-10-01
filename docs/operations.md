# Operations

## Startup

Activate the project virtual environment and validate configuration:

```bash
source .venv/bin/activate
vsglasses --check-config
vsglasses --config configs/raspberry_pi.toml
```

For interactive development with a live window, use the single root entry
point:

```bash
python main.py
```

On macOS, camera capture, inference, and the Cocoa-backed OpenCV window run on
the main thread because OpenCV's AVFoundation capture can crash when its native
camera delegate is split from GUI lifecycle across Python threads. The
Raspberry Pi service runtime retains bounded background acquisition.

The default camera is optional. If OpenCV is missing, the USB camera is absent,
or a read fails, acquisition reports a recovering state and retries with bounded
exponential backoff. Other application capabilities remain available.

Use `Ctrl+C` or send `SIGTERM` for a clean shutdown. Camera resources and waiting
frame consumers are released during shutdown.

## Diagnostics

Use `--run-seconds N` for bounded diagnostic runs. Acquisition metrics include
successful frames, open failures, read failures, reconnects, overwritten stale
frames, and effective average FPS.

Raw frames are held only in a one-item in-memory latest-frame buffer. They are
not persisted by the acquisition service.

## Validation

Run normal tests with `pytest`. Tests marked `hardware` require connected
peripherals and should be run separately on the Raspberry Pi. Before completing
Phase 1 validation, perform camera disconnect/reconnect testing and an eight-hour
soak test on the target Raspberry Pi. The exact commands and evidence checklist
are documented in `phase-1-validation.md`.
