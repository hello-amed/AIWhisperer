import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pynput import keyboard
from parakeet_mlx import from_pretrained
import os
import sys

# Ensure the script uses the virtual environment libraries
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_site_packages = os.path.join(script_dir, ".venv", "lib", "python3.9", "site-packages")
sys.path.append(venv_site_packages)

# 1. Initialize the MLX-optimized Parakeet model
print("🦜 AiWhisperer is waking up...")
# The first run will automatically download the model (approx. 600MB)
model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")

FS = 16000
recording = []
is_recording = False

def callback(indata, frames, time, status):
    if is_recording:
        recording.append(indata.copy())

def on_press(key):
    global is_recording, recording
    # Listen specifically for the Right Command key
    if key == keyboard.Key.cmd_r and not is_recording:
        recording = []
        is_recording = True
        print("🎤 [AiWhisperer] Listening... (Hold Right Cmd)")

def on_release(key):
    global is_recording, recording
    if key == keyboard.Key.cmd_r and is_recording:
        is_recording = False
        print("⚡ [AiWhisperer] Transcribing...")
        
        # Save audio to a temp file
        audio_data = np.concatenate(recording, axis=0)
        temp_path = "/tmp/ai_whisperer_audio.wav"
        write(temp_path, FS, audio_data)
        
        # Transcribe using the MLX model
        result = model.transcribe(temp_path)
        
        # Type the text into your active window
        controller = keyboard.Controller()
        controller.type(result.text)
        print(f"✅ [AiWhisperer] Inserted text.")

# Background audio and keyboard listener
with sd.InputStream(samplerate=FS, channels=1, callback=callback):
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        print("🔥 AiWhisperer Active. HOLD 'Right Command' to talk.")
        listener.join()