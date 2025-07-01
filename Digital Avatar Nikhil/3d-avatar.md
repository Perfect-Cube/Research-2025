## Project: 3D Voice-Enabled Avatar on Jetson Nano

### Directory Structure
```
3d_avatar_project/
├── models/
│   └── avatar.egg      # Low-poly Panda3D model with jaw or mouth joint
├── main.py             # Entry point
├── avatar.py           # Avatar rendering & lip-sync
├── speech_io.py        # Speech recognition & TTS
├── requirements.txt    # Python dependencies
└── README.md           # Project overview
```

---
### requirements.txt
```
# 3D Rendering
panda3d

# Speech Recognition
vosk
sounddevice

# Text-to-Speech
pyttsx3

# Audio processing
numpy
```

---
### main.py
```python
from direct.showbase.ShowBase import ShowBase
from panda3d.core import ClockObject
from avatar import AvatarController
from speech_io import SpeechIO

class App(ShowBase):
    def __init__(self):
        super().__init__()
        # Limit frame rate to reduce CPU/GPU load
        globalClock = ClockObject.getGlobalClock()
        globalClock.setMode(ClockObject.MLimited)
        globalClock.setFrameRate(30)

        # Initialize avatar
        self.avatar = AvatarController(render)

        # Setup speech I/O
        self.speech = SpeechIO(on_text=self.on_text_recognized)
        self.speech.start_listening()

        # Update task for lip-sync
        self.taskMgr.add(self.update, 'update')

    def on_text_recognized(self, text):
        # Simple echo TTS; you can integrate LLM or custom response
        response = f"You said: {text}"
        self.speech.speak(response)
        self.avatar.mouth.queue_text(response)

    def update(self, task):
        dt = globalClock.getDt()
        self.avatar.update(dt)
        return task.cont

if __name__ == '__main__':
    app = App()
    app.run()
```

---
### avatar.py
```python
from direct.actor.Actor import Actor
import numpy as np

# Map phonemes to jaw angles (simple viseme mapping)
PHONEME_TO_ANGLE = {
    'A': 15, 'B': 5, 'C': 5, 'D': 5, 'E': 10, 'F': 0, 'G': 5,
    'default': 2
}

class AvatarController:
    def __init__(self, parent):
        # Load a simple egg model with a mouth joint named 'jaw'
        self.actor = Actor("models/avatar.egg", {})
        self.actor.reparentTo(parent)
        self.jaw = self.actor.controlJoint(None, 'modelRoot', 'jaw')
        self.current_angle = 0
        self.target_angle = 0
        self.text_queue = []

    def queue_text(self, text):
        # Split text into phonemes (naive split)
        phonemes = list(text.upper())
        self.text_queue += phonemes

    def update(self, dt):
        if self.text_queue:
            # Pop next phoneme
            phoneme = self.text_queue.pop(0)
            self.target_angle = PHONEME_TO_ANGLE.get(phoneme, PHONEME_TO_ANGLE['default'])
        else:
            self.target_angle = PHONEME_TO_ANGLE['default']

        # Smooth interpolation
        self.current_angle += (self.target_angle - self.current_angle) * min(dt*10, 1)
        self.jaw.setHpr(self.current_angle, 0, 0)
```

---
### speech_io.py
```python
import threading
import queue
import sounddevice as sd
import pyttsx3
from vosk import Model, KaldiRecognizer
import json

class SpeechIO:
    def __init__(self, on_text):
        self.on_text = on_text
        self.rec_queue = queue.Queue()
        self.model = Model("models/vosk-model-small-en-us")
        self.rec = KaldiRecognizer(self.model, 16000)
        self.tts = pyttsx3.init()
        self.listening = False

    def start_listening(self):
        self.listening = True
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def _listen_loop(self):
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1) as stream:
            while self.listening:
                data = stream.read(4000)[0]
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    text = result.get('text', '')
                    if text:
                        self.on_text(text)

    def speak(self, text):
        threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text):
        self.tts.say(text)
        self.tts.runAndWait()
