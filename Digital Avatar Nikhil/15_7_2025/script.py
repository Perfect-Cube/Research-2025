import os
import groq
from playsound import playsound
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

# --- 1. CONFIGURATION ---
# Ensure your API key is set as an environment variable
GROQ_API_KEY = "key"
if not GROQ_API_KEY:
    raise ValueError("FATAL: GROQ_API_KEY environment variable not set.")

# Audio recording settings
SAMPLE_RATE = 44100  # Sample rate in Hz
RECORD_SECONDS = 5   # Duration of recording
USER_AUDIO_FILENAME = "temp_user_audio.wav" # Temporary file for user's speech

# Path to the folder containing the avatar's pre-recorded audio files
AVATAR_AUDIO_DIR = "audio_files"


# --- 2. DEFINE ALL NECESSARY FUNCTIONS ---

def record_audio_from_mic():
    """Records audio from the default microphone and saves it as a WAV file."""
    print(f"\n----- Recording for {RECORD_SECONDS} seconds... Speak now! -----")

    # Record audio using sounddevice, which returns a NumPy array
    recording = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')

    # Wait until the recording is finished
    sd.wait()

    # Save the NumPy array as a WAV file using SciPy
    write(USER_AUDIO_FILENAME, SAMPLE_RATE, recording)

    print("----- Finished recording. -----")
    return USER_AUDIO_FILENAME


def transcribe_user_audio(file_path):
    """Transcribes audio to text using the Groq API."""
    print("Transcribing audio...")
    client = groq.Groq(api_key=GROQ_API_KEY)
    with open(file_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(file_path), file.read()),
            model="whisper-large-v3"
        )
    return transcription.text


def get_user_intent(transcribed_text):
    """Parses text to determine user intent based on keywords."""
    text_lower = transcribed_text.lower()
    print(f"Analyzing text for intent: '{text_lower}'")
    if any(word in text_lower for word in ["hi", "hello", "how are you", "hey"]): return "GREETING"
    if any(word in text_lower for word in ["what can you do", "help", "tasks", "capabilities"]): return "CAPABILITIES"
    if any(word in text_lower for word in ["interesting", "fact", "tell me something"]): return "INTERESTING_FACT"
    if any(word in text_lower for word in ["sing", "song"]): return "SINGING"
    return "FALLBACK"


def play_avatar_response(intent):
    """Plays the correct pre-recorded audio file based on the intent."""
    avatar_audio_map = {
        "GREETING": "audio_files/1.wav",
        "CAPABILITIES": "audio_files/2.wav",
        "INTERESTING_FACT": "audio_files/3.wav",
        "SINGING": "audio_files/4.wav",
        "FALLBACK": "1.wav"
    }
    response_filename = avatar_audio_map.get(intent)
    
    # Construct the full path to the audio file
    audio_file_path = "D:/Volkswagen/Kokoro/" + response_filename

    if os.path.exists(audio_file_path):
        print(f"Intent detected: '{intent}'. Playing response: {audio_file_path}")
        try:
            playsound(audio_file_path)
        except Exception as e:
            print(f"Error playing sound: {e}")
    else:
        print(f"ERROR: The audio file '{audio_file_path}' was not found.")


# --- 3. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("--- Starting Interactive Avatar Session (Press Ctrl+C to exit) ---")
    try:
        # Step 1: Record audio from the microphone
        user_audio_file = record_audio_from_mic()

        # Step 2: Transcribe the recorded audio
        user_text = transcribe_user_audio(user_audio_file)
        print(f"\nUser said: '{user_text}'")

        # Step 3: Determine the user's intent
        detected_intent = get_user_intent(user_text)

        # Step 4: Play the avatar's corresponding response
        play_avatar_response(detected_intent)

    except KeyboardInterrupt:
        print("\nSession ended by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # Clean up the temporary user audio file
        if os.path.exists(USER_AUDIO_FILENAME):
            os.remove(USER_AUDIO_FILENAME)
            print(f"\nCleaned up temporary file: {USER_AUDIO_FILENAME}")
