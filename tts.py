"""Local Piper TTS adapter."""
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

    def speak(self, text: str) -> None:
        if not os.path.isfile(self.model):
            print(f"TTS unavailable; set PIPER_MODEL to a Piper .onnx voice: {text}")
            return
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            subprocess.run(self.command + ["--model", self.model,
                            "--output_file", audio_file.name],
                           input=text.encode(), check=False)
            player = (shutil.which("afplay") or shutil.which("aplay")
                      or shutil.which("paplay"))
            if player:
                subprocess.run(([player, "-D", "plughw:MAX98357A,0", audio_file.name] if player.endswith("aplay") else [player, audio_file.name]), check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                print(f"Audio player unavailable; generated: {audio_file.name}")
