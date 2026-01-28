import os
import sys

# 1. FORCE UTF-8 ENCODING 
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"
# 2. ENSURE FFMPEG IS ON PATH 
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

import rumps
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pynput import keyboard
from parakeet_mlx import from_pretrained

# Global model variable (loaded async)
model = None

class AiWhispererApp(rumps.App):
    def __init__(self):
        super(AiWhispererApp, self).__init__("⏳", title=None)
        self.menu = ["Status: Loading Model...", None, "Quit"]
        self.fs = 16000
        self.recording = []
        self.is_recording = False
        
        # Start loading model in background
        import threading
        threading.Thread(target=self.load_model, daemon=True).start()
        
        # Start keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        # LOGGING SETUP
        self.log_file = os.path.expanduser("~/ai_whisperer.log")
        with open(self.log_file, "w") as f:
            f.write("--- App Started ---\n")
            
    def log(self, message):
        timestamp = os.popen('date "+%H:%M:%S"').read().strip()
        entry = f"[{timestamp}] {message}"
        print(message) # Still print to stdout
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(entry + "\n")

    def load_model(self):
        global model
        try:
            self.log("🦜 Downloading/Loading model...")
            # Using the user-requested MLX optimized model
            model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3") 
            
            self.title = "🦜"
            self.menu["Status: Loading Model..."].title = "Status: Ready"
            self.log("✅ Model Loaded Successfully")
        except Exception as e:
            self.title = "⚠️"
            self.menu["Status: Loading Model..."].title = f"Error: {str(e)[:20]}"
            self.log(f"Error loading model: {e}")

    def on_press(self, key):
        if key == keyboard.Key.cmd_r and not self.is_recording:
            if model is None:
                self.log("Wait for model to load...")
                return
            self.recording = []
            self.is_recording = True
            self.title = "🔴" 
            self.log("🎤 Recording started...")

    def on_release(self, key):
        if key == keyboard.Key.cmd_r and self.is_recording:
            self.is_recording = False
            self.title = "⚡"
            self.log("🛑 Recording stopped.")
            
            # Save and Transcribe
            if not self.recording:
                self.log("⚠️ No audio data recorded.")
                self.title = "🦜"
                return

            audio_data = np.concatenate(self.recording, axis=0)
            max_amp = np.max(np.abs(audio_data))
            self.log(f"📊 Audio stats: Shape={audio_data.shape}, Max Amp={max_amp:.4f}")
            
            temp_path = os.path.join(os.getenv("TMPDIR", "/tmp"), "whisper.wav")
            write(temp_path, self.fs, audio_data)
            self.log(f"💾 Saved audio to {temp_path} ({os.path.getsize(temp_path)} bytes)")
            
            try:
                if model:
                    self.log("🧠 Analyzing audio...")
                    result = model.transcribe(temp_path)
                    self.log(f"📝 Transcribed: '{result.text}'")
                    
                    if result.text and result.text.strip():
                        self.log("⌨️ Typing...")
                        keyboard.Controller().type(result.text)
                    else:
                         self.log("⚠️ Transcription was empty.")
                else:
                    self.log("Model not loaded yet.")
            except Exception as e:
                self.log(f"Error: {e}")
            
            self.title = "🦜"

    def record_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio Status: {status}")
        if self.is_recording:
            self.recording.append(indata.copy())

if __name__ == "__main__":
    app = AiWhispererApp()
    with sd.InputStream(samplerate=16000, channels=1, callback=app.record_callback):
        app.run()