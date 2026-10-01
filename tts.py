"""Local Piper TTS adapter."""
import json
import os
import shutil
import subprocess
import sys
import tempfile


class PiperSpeech:
    def __init__(self) -> None:
        executable = shutil.which("piper")
        self.command = [executable] if executable else [sys.executable, "-m", "piper"]
        self.model = os.environ.get("PIPER_MODEL", "models/en_US-lessac-medium.onnx")
        self.volume = os.environ.get("PIPER_VOLUME_BOOST", "3.0")
        self.sample_rate = self._sample_rate()

    def _sample_rate(self) -> int:
        config_path = os.environ.get("PIPER_CONFIG", f"{self.model}.json")
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                return int(json.load(config_file).get("audio", {}).get("sample_rate", 22050))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return 22050

    def _speak_streaming_raw(self, text: str, player: str) -> bool:
        if not player.endswith("aplay"):
            return False
        piper_cmd = self.command + [
            "--model", self.model,
            "--output-raw",
            "--volume", self.volume,
        ]
        audio_device = os.environ.get("PIPER_AUDIO_DEVICE", "plughw:MAX98357A,0")
        player_cmd = [
            player,
            "-q",
            "-D", audio_device,
            "-t", "raw",
            "-f", "S16_LE",
            "-r", str(self.sample_rate),
            "-c", "1",
            "--buffer-time", os.environ.get("PIPER_APLAY_BUFFER_TIME", "50000"),
            "--period-time", os.environ.get("PIPER_APLAY_PERIOD_TIME", "10000"),
        ]
        piper = subprocess.Popen(
            piper_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        player_proc = subprocess.Popen(
            player_cmd,
            stdin=piper.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if piper.stdout is not None:
            piper.stdout.close()
        if piper.stdin is not None:
            piper.stdin.write(text.encode())
            piper.stdin.close()
        piper.wait()
        player_proc.wait()
        return piper.returncode == 0 and player_proc.returncode == 0

    def speak(self, text: str) -> None:
        if not os.path.isfile(self.model):
            print(f"TTS unavailable; set PIPER_MODEL to a Piper .onnx voice: {text}")
            return
        player = (shutil.which("afplay") or shutil.which("aplay")
                  or shutil.which("paplay"))
        if player and self._speak_streaming_raw(text, player):
            return
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            subprocess.run(self.command + ["--model", self.model,
                            "--output_file", audio_file.name,
                            "--volume", self.volume],
                           input=text.encode(), check=False)
            if player:
                subprocess.run(([player, "-D", os.environ.get("PIPER_AUDIO_DEVICE", "plughw:MAX98357A,0"), audio_file.name] if player.endswith("aplay") else [player, audio_file.name]), check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                print(f"Audio player unavailable; generated: {audio_file.name}")
