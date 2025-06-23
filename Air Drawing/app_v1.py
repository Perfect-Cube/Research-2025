# ==============================================================================
#                      AI Drawing Assistant for Google Colab (v6 - Robust Initialization)
#
# This single script will:
# 1. Install all necessary libraries.
# 2. Set up API keys and initialize AI models (LLaVA & Stable Diffusion).
# 3. Create an interactive UI to draw with your hand via webcam.
#
# INSTRUCTIONS:
# 1. Add your Groq API Key to Colab's Secrets (Key icon 🔑 on the left)
#    - Name: GROQ_API_KEY
#    - Value: your_api_key_here
# 2. **IMPORTANT**: Use a GPU runtime for this to work well.
#    Go to `Runtime` -> `Change runtime type` -> Select `T4 GPU`.
# 3. Run this entire cell.
# 4. Grant webcam access when your browser asks.
# ==============================================================================


# --- 1. INSTALLATION ---
print("⏳ Step 1/5: Installing necessary libraries...")
!pip install opencv-python-headless mediapipe groq torch diffusers transformers accelerate ipywidgets -q
print("✅ Installation complete.")


# --- 2. IMPORTS AND API KEY SETUP ---
print("\n⏳ Step 2/5: Importing libraries and setting up API Key...")
import cv2
import numpy as np
import mediapipe as mp
import os
import base64
from groq import Groq
import torch
from diffusers import StableDiffusionPipeline
import time
from google.colab import output, userdata
from IPython.display import display, Javascript
import ipywidgets as widgets

try:
    # GROQ_API_KEY = userdata.get('GROQ_API_KEY')
    print("✅ Groq API Key loaded successfully.")
except userdata.SecretNotFoundError:
    print("🛑 ERROR: 'GROQ_API_KEY' not found in secrets.")
    raise
except Exception as e:
    print(f"An error occurred loading the key: {e}")
    raise


# --- 3. INITIALIZE AI MODELS ---
print("\n⏳ Step 3/5: Initializing AI Models (this will take a few minutes)...")

# Configuration
WIDTH, HEIGHT = 640, 480
DRAW_COLOR = (255, 255, 255)
DRAW_THICKNESS = 15

groq_client = Groq(api_key=GROQ_API_KEY)
# LLAVA_MODEL = "llava-v1.5-7b-vision"
print("  - Groq client initialized.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  - Using device: {device}")
if device == "cpu": print("  - ⚠️ WARNING: CPU mode will be very slow for image generation. Consider switching to a GPU runtime.")
# model_id = "runwayml/stable-diffusion-v1-5"
sd_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
sd_pipe = sd_pipe.to(device)
print("  - Stable Diffusion pipeline ready.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
print("  - MediaPipe Hands initialized.")
print("✅ All models initialized!")


# --- 4. HELPER FUNCTIONS ---
print("\n⏳ Step 4/5: Defining helper functions...")

# State Variables
canvas = np.zeros((HEIGHT, WIDTH, 3), dtype="uint8")
prev_x, prev_y = None, None

# Camera Functions for Colab
def capture_frame(width=WIDTH, height=HEIGHT):
    # This single function now handles both initialization and frame capturing,
    # eliminating the race condition.
    js_code = f'''
    async function captureFrame() {{
      // 1. Check if the video stream is already running.
      if (!window.stream || !window.video) {{
        // If not, create and initialize it.
        const div = document.createElement('div');
        const video = document.createElement('video');
        video.style.display = 'none'; // We don't need to see the raw feed.
        const stream = await navigator.mediaDevices.getUserMedia({{video: true}});
        
        document.body.appendChild(div);
        div.appendChild(video);
        video.srcObject = stream;
        
        // Store the stream and video elements globally for reuse.
        window.stream = stream;
        window.video = video;

        // Wait for the video to be ready to play.
        await new Promise((resolve) => video.onloadedmetadata = resolve);
        await video.play();
        
        google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
      }}

      // 2. Now that we're sure the video is ready, capture a frame.
      const canvas = document.createElement('canvas');
      canvas.width = {width};
      canvas.height = {height};
      canvas.getContext('2d').drawImage(window.video, 0, 0, {width}, {height});
      return canvas.toDataURL('image/jpeg');
    }}
    captureFrame();
  '''
    data_url = output.eval_js(js_code)
    if data_url is None: return None
    img_bytes = base64.b64decode(data_url.split(',')[1])
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_np, flags=1)
    return frame

def stop_webcam():
    js = Javascript('''
        if (window.stream) {
            window.stream.getTracks().forEach(track => track.stop());
        }
        if (window.video) {
            window.video.remove();
        }
    ''')
    try:
      display(js)
    except Exception:
      pass

# AI & Image Functions
def encode_image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def get_drawing_description(image_base64):
    try:
        chat_completion = groq_client.chat.completions.create(model=LLAVA_MODEL, messages=[{"role": "user", "content": [{"type": "text", "text": "This is a crude line drawing on a black background. What is it? Describe it in a short, simple phrase for an image generator (e.g., 'a house with a sun', 'a smiling cat')."},{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]}], max_tokens=50)
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling Groq API: {e}"); return None

def generate_image_from_prompt(prompt):
    enhanced_prompt = f"award-winning photo of {prompt}, 4k, photorealistic, high detail, masterpiece"
    image = sd_pipe(enhanced_prompt).images[0]
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def process_frame(frame):
    global canvas, prev_x, prev_y
    if frame is None: return None, None
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    is_drawing = False
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            if index_tip.y < middle_tip.y - 0.05:
                is_drawing = True
                x, y = int(index_tip.x * WIDTH), int(index_tip.y * HEIGHT)
                if prev_x is None: prev_x, prev_y = x, y
                cv2.line(canvas, (prev_x, prev_y), (x, y), DRAW_COLOR, DRAW_THICKNESS)
                prev_x, prev_y = x, y
    if not is_drawing: prev_x, prev_y = None, None
    return frame, canvas
print("✅ Helper functions defined.")


# --- 5. RUN THE INTERACTIVE APPLICATION ---
print("\n🚀 Step 5/5: Starting the application!")
print("================================================================")
print("The first frame will take a moment to initialize the webcam.")
print("Raise your index finger to draw. Lower it to move without drawing.")

# UI Widgets
predict_button = widgets.Button(description="Predict & Generate", button_style='success', icon='magic')
erase_button = widgets.Button(description="Erase Canvas", button_style='warning', icon='trash-alt')
stop_button = widgets.Button(description="Stop Session", button_style='danger', icon='stop-circle')
ui_output = widgets.Output()
image_output = widgets.Output()
final_image_output = widgets.Output()

is_running = True

def on_predict_clicked(b):
    with final_image_output:
        final_image_output.clear_output(wait=True)
        print("⏳ Analyzing drawing with LLaVA...")
        if np.sum(canvas) == 0: print("🎨 Canvas is empty. Draw something first!"); return
        drawing_base64 = encode_image_to_base64(canvas)
        description = get_drawing_description(drawing_base64)
        if description:
            print(f"🤖 LLaVA says it's a: '{description}'")
            print("⏳ Generating image with Stable Diffusion...")
            generated_image = generate_image_from_prompt(description)
            print("✅ Image Generated! Displaying below.")
            _, img_encoded = cv2.imencode('.png', generated_image)
            display(widgets.Image(value=img_encoded.tobytes(), format='png', width=512, height=512))
        else: print("😞 Could not get a description for the drawing.")

def on_erase_clicked(b):
    global canvas
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype="uint8")
    with final_image_output: final_image_output.clear_output(wait=True)
    print("✨ Canvas cleared!")

def on_stop_clicked(b):
    global is_running
    is_running = False

predict_button.on_click(on_predict_clicked)
erase_button.on_click(on_erase_clicked)
stop_button.on_click(on_stop_clicked)

button_box = widgets.HBox([predict_button, erase_button, stop_button])
display(ui_output, button_box, image_output, final_image_output)

try:
    with ui_output: print("Please grant webcam access when prompted by your browser...")
    while is_running:
        frame = capture_frame(WIDTH, HEIGHT)
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Clear the "Please grant access" message on the first successful frame
        with ui_output: ui_output.clear_output()

        processed_frame, current_canvas = process_frame(frame)
        if processed_frame is None:
            time.sleep(0.1)
            continue
        
        combined_display = np.hstack([processed_frame, current_canvas])
        _, img_encoded = cv2.imencode('.png', combined_display)
        with image_output:
            image_output.clear_output(wait=True)
            display(widgets.Image(value=img_encoded.tobytes(), format='png'))
        
        time.sleep(0.01)
except Exception as e:
    if is_running: print(f"An error occurred in the main loop: {e}")
finally:
    is_running = False
    stop_webcam()
    with ui_output:
        ui_output.clear_output(wait=True)
        print("🛑 Session stopped. Run the cell again to restart.")
