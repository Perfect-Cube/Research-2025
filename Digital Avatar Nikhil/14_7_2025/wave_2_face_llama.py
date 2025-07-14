# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import pygame
import warnings
import time
import pyaudio
import wave
from threading import Thread
from fuzzywuzzy import process
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

# --- NEW: Groq and Audio Recording Setup ---
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

def record_audio():
    """Records audio from the microphone until Enter is pressed."""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    frames = []
    print("Recording... Press Enter to stop.")
    try:
        while True:
            # A non-blocking way to check for user input would be more robust,
            # but for simplicity, we'll just let this loop run. The user needs to press Enter in the console.
            data = stream.read(CHUNK)
            frames.append(data)
            # This check is a placeholder; in a real app, you'd use a different method to stop.
            # For this script, we'll rely on the user pressing Enter in the main loop.
    except KeyboardInterrupt:
        pass # This allows stopping with Ctrl+C if needed

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(TEMP_WAV_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("Recording finished.")

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

def find_relevant_answer_with_groq(question, script_content):
    """Uses Groq's language model to find the most relevant answer from the script."""
    print("Finding relevant answer...")
    
    # Prepare the content for the prompt
    script_for_prompt = "\n".join([f"- {text}" for text in script_content])

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. The user will ask a question, and you must choose the most relevant paragraph from the following list. Respond with ONLY the text of the chosen paragraph and nothing else.\n\n{script_for_prompt}",
                },
                {
                    "role": "user",
                    "content": question,
                }
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error with Groq chat completion: {e}")
        return None

if __name__ == "__main__":
    
    initialize_directories()
    wav_input_folder = os.path.join(os.getcwd(), 'wav_input')
    ensure_wav_input_folder_exists(wav_input_folder)
    py_face = initialize_py_face()
    socket_connection = create_socket_connection()
    default_animation_thread = Thread(target=default_animation_loop, args=(py_face,))
    default_animation_thread.start()

    # --- MODIFIED: Script Content and Main Loop ---
    
    # IMPORTANT: Populate this with your paragraph text and corresponding audio file names.
    script_data = {
        "This is the first paragraph of your presentation, covering the introduction.": "intro.wav",
        "This second paragraph explains the core problem we are addressing.": "problem_statement.wav",
        "The third part of our talk details our innovative solution.": "solution.wav",
        "Here in the fourth paragraph, we discuss the market analysis and opportunity.": "market_analysis.wav",
        "And finally, the fifth paragraph concludes our presentation with a call to action.": "conclusion.wav",
        # Add all 10 of your paragraphs and audio files here
    }

    try:
        while True:
            input("\nPress Enter to start recording your question...")
            
            # For simplicity, this is a blocking call. Speak and then return to the console to press Enter again.
            # A more advanced implementation might use a separate thread for input.
            # We will use KeyboardInterrupt (Ctrl+C) to stop recording for this example.
            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            frames = []
            print("Recording... Press Ctrl+C to stop.")
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
                print(f"You said: {transcribed_text}")
                
                # Get the relevant paragraph text from Groq
                answer_text = find_relevant_answer_with_groq(transcribed_text, script_data.keys())

                if answer_text:
                    print(f"Found relevant answer: {answer_text}")
                    
                    # Use fuzzy matching to find the best-matching key in our script_data
                    best_match, score = process.extractOne(answer_text, script_data.keys())
                    
                    if score > 80: # Using a confidence threshold of 80
                        audio_file_to_play = script_data[best_match]
                        print(f"Playing corresponding audio: {audio_file_to_play}")
                        
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
                    else:
                        print(f"Could not find a confident match for the answer. (Best match: '{best_match}' with score {score})")

            if os.path.exists(TEMP_WAV_FILE):
                os.remove(TEMP_WAV_FILE) # Clean up the temporary file

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
