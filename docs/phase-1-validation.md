# Phase 1 target-hardware validation

The software implementation is complete when normal automated checks pass.
Phase 1 is accepted for Raspberry Pi deployment only after the following
physical tests are recorded.

## USB camera smoke test

```bash
source .venv/bin/activate
VSGLASSES_RUN_HARDWARE_TESTS=1 pytest tests/hardware/test_usb_camera.py
```

## Disconnect and reconnect

1. Start `vsglasses --config configs/raspberry_pi.toml`.
2. Confirm frames are acquired.
3. Disconnect the webcam.
4. Confirm a recovering health state and bounded retry behavior.
5. Reconnect the webcam.
6. Confirm acquisition resumes without restarting the process.
7. Stop with `Ctrl+C` and confirm the device is released.

## Soak test

Run the helper for eight hours:

```bash
scripts/camera_soak_test.sh 28800
```

During and after the run, record:

- successful frames and effective FPS;
- open and read failures;
- reconnect count;
- process RAM and CPU;
- Raspberry Pi temperature;
- unexpected termination or unbounded resource growth.

Increase to 24 hours after the eight-hour run succeeds.
