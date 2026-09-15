# Raspberry Pi runtime

The Pi deployment uses one OpenCV capture (`/dev/video0`) and one YOLO11n
instance. The Flask dashboard only reads the newest annotated JPEG and status
snapshot published by the perception loop.
The system-status panel reports CPU temperature and the Pi 5 PMIC's real
`EXT5V_V` input-rail measurement every five seconds. Power health comes from
the firmware undervoltage/throttling flags. The IMU panel reads an MPU6050 on
I²C bus 1 at 20 Hz and reports disconnected status when `0x68`/`0x69` is absent.

## Start

```bash
cd /home/rohan/VSGlasses-pi
./run_pi.sh
```

Open `http://192.168.4.115:3000` from a device on the same Eero network.

For camera, YOLO, and dashboard operation without voice:

```bash
./run_pi.sh --no-voice
```

## Voice commands

The Logitech C270 microphone is used for offline Vosk recognition. Say
`hey glasses`, then a command; recognised text and the local response appear
in the dashboard. Audio playback is disabled by default because no speaker is
connected. Add `--tts` only after connecting an output device.

Text replies use local Ollama model `qwen2.5:0.5b-instruct`; this is text-only
and unrelated to the removed room-detection model.

The Pi defaults to `--imgsz 480`, which measured about 0.20 seconds per warmed
YOLO inference. Use `--imgsz 640` for more detail or `--imgsz 320` for speed.

For an SSH tunnel:

```bash
ssh -L 3000:127.0.0.1:3000 rohan@192.168.4.115
```

Then open `http://localhost:3000` on the Mac.

The runtime is fully local after its Python dependencies have been installed.
Qwen room inference was intentionally removed because the 4B vision model is
too resource-intensive for this 4 GB Raspberry Pi.
