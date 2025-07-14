# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import pygame
import warnings
import time
import pyaudio
import wave
import re  # Imported for regular expressions
from threading import Thread
from groq import Groq

# --- Local Imports from your project ---
from livelink.connect.livelink_init import create_socket_connection, initialize_py_face
from livelink.animations.default_animation import default_animation_loop, stop_default_animation
from utils.files.file_utils import initialize_directories, ensure_wav_input_folder_exists
from utils.audio_face_workers import process_wav_file
from utils.emote_sender.send_emote import EmoteConnect

warnings.filterwarnings(
    "ignore", 
    message="Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work"
)

ENABLE_EMOTE_CALLS = False

# --- Groq and Audio Recording Setup ---
try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    print("Please make sure you have set the GROQ_API_KEY environment variable.")
    exit()

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
TEMP_WAV_FILE = "temp_recording.wav"

def transcribe_with_groq():
    """Transcribes the recorded audio file using Groq's Whisper API."""
    print("Transcribing audio...")
    try:
        with open(TEMP_WAV_FILE, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(TEMP_WAV_FILE, file.read()),
                model="whisper-large-v3",
            )
        return transcription.text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

# --- NEW: Rule-based answer finding using if/else and regex ---
def find_answer_with_rules(transcribed_text):
    """
    Finds the correct audio file to play based on keywords in the text.
    
    Args:
        transcribed_text (str): The text from the user's speech.

    Returns:
        str: The filename of the audio to play, or a fallback filename.
    """
    print(f"Searching for keywords in: '{transcribed_text}'")

    # --- CUSTOMIZE YOUR KEYWORDS HERE ---
    # Use `|` to mean "OR". The `r` before the string is important.
    
    # Keywords for your introduction audio
    if re.search(r'introduce|who are you|tell me about yourself|what is this', transcribed_text, re.IGNORECASE):
        return 'intro.wav'
    
    # Keywords for your problem statement audio
    elif re.search(r'problem|issue|challenge|what do you solve', transcribed_text, re.IGNORECASE):
        return 'problem_statement.wav'
        
    # Keywords for your solution audio
    elif re.search(r'solution|how do you fix|what is the answer|proposal', transcribed_text, re.IGNORECASE):
        return 'solution.wav'
        
    # Keywords for your market analysis audio
    elif re.search(r'market|opportunity|business case|customers', transcribed_text, re.IGNORECASE):
        return 'market_analysis.wav'
        
    # Keywords for your conclusion audio
    elif re.search(r'conclude|summary|call to action|wrap up|final thoughts', transcribed_text, re.IGNORECASE):
        return 'conclusion.wav'

    # --- Add more elif blocks for your other 5 paragraphs ---
    # elif re.search(r'keyword1|keyword2', transcribed_text, re.IGNORECASE):
    #     return 'your_audio_file_6.wav'

    else:
        # This is the fallback if no keywords are matched.
        # You should create a "fallback.wav" file with a message like "I'm not sure how to answer that."
        print("No keyword match found. Using fallback response.")
        return 'fallback.wav'


if __name__ == "__main__":
    
    initialize_directories()
    wav_input_folder = os.path.join(os.getcwd(), 'wav_input')
    ensure_wav_input_folder_exists(wav_input_folder)
    py_face = initialize_py_face()
    socket_connection = create_socket_connection()
    default_animation_thread = Thread(target=default_animation_loop, args=(py_face,))
    default_animation_thread.start()

    try:
        while True:
            input("\nPress Enter to start recording your question...")
            
            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            frames = []
            print("Recording... Press Ctrl+C in this terminal to stop.")
            try:
                while True:
                    data = stream.read(CHUNK)
                    frames.append(data)
            except KeyboardInterrupt:
                pass
            
            print("Recording stopped.")
            stream.stop_stream()
            stream.close()
            p.terminate()

            wf = wave.open(TEMP_WAV_FILE, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            transcribed_text = transcribe_with_groq()

            if transcribed_text:
                # Get the relevant audio file using our new rule-based function
                audio_file_to_play = find_answer_with_rules(transcribed_text)
                
                print(f"Action: Playing corresponding audio -> {audio_file_to_play}")
                selected_file = os.path.join(wav_input_folder, audio_file_to_play)

                if os.path.exists(selected_file):
                    if ENABLE_EMOTE_CALLS:
                        EmoteConnect.send_emote("startspeaking")
                    
                    try:
                        process_wav_file(selected_file, py_face, socket_connection, default_animation_thread)
                    finally:
                        if ENABLE_EMOTE_CALLS:
                            EmoteConnect.send_emote("stopspeaking")
                else:
                    print(f"Error: Could not find the audio file '{audio_file_to_play}' in the 'wav_input' folder.")

            if os.path.exists(TEMP_WAV_FILE):
                os.remove(TEMP_WAV_FILE) # Clean up

            user_choice = input("Ask another question? (y/n): ").strip().lower()
            if user_choice != 'y':
                break

    finally:
        stop_default_animation.set()
        if default_animation_thread:
            default_animation_thread.join()
        pygame.quit()
        if socket_connection:
            socket_connection.close()
        print("Cleanup complete. Program terminated.")
