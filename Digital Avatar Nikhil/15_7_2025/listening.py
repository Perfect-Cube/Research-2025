import webrtcvad
import pyaudio
import collections
import time

# --- Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # VAD only accepts 8000, 16000, 32000, or 48000 Hz
FRAME_DURATION_MS = 30  # VAD only accepts 10, 20, or 30 ms
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)
SILENCE_DURATION_SECS = 5

# --- VAD and Audio Stream Initialization ---
vad = webrtcvad.Vad(3)  # Set aggressiveness mode (0 to 3)
audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=FRAME_SIZE)

print("Listening for silence...")

silence_start_time = None
silence_detected = False

try:
    while True:
        frame = stream.read(FRAME_SIZE)
        is_speech = vad.is_speech(frame, RATE)

        if not is_speech:
            if silence_start_time is None:
                silence_start_time = time.time()

            if not silence_detected and (time.time() - silence_start_time) >= SILENCE_DURATION_SECS:
                print(True)
                silence_detected = True
        else:
            if silence_start_time is not None:
                if silence_detected:
                    print(False)
                silence_start_time = None
                silence_detected = False

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    # --- Cleanup ---
    stream.stop_stream()
    stream.close()
    audio.terminate()```

### How the Code Works:

1.  **Configuration**: We define the audio format, channels, sample rate, and frame duration. `webrtcvad` has specific requirements for the sample rate and frame duration. We also set the desired silence duration to 5 seconds.
2.  **Initialization**: We initialize the `webrtcvad` object and open a `pyaudio` stream to capture audio from the microphone. The `Vad` object's aggressiveness mode can be set from 0 (least aggressive) to 3 (most aggressive) in filtering out non-speech.
3.  **Listening Loop**: The code continuously reads audio in small chunks (frames) from the microphone.
4.  **Voice Activity Detection**: Each frame is passed to `vad.is_speech()` to determine if it contains speech.
5.  **Silence Tracking**:
    *   If a frame is determined to be non-speech, we check if a `silence_start_time` has been recorded. If not, we record the current time.
    *   We then check if the duration of the current silence has exceeded our `SILENCE_DURATION_SECS` and if we haven't already printed `True` for this silence period. If both conditions are met, it prints `True` and sets a flag `silence_detected` to `True`.
    *   If a frame contains speech, we reset the `silence_start_time` and `silence_detected` flag. If a 5-second silence was previously detected, it will print `False` to indicate the end of that silent period.
6.  **Cleanup**: When you stop the script (e.g., with Ctrl+C), it gracefully closes the audio stream and terminates the `pyaudio` instance.
