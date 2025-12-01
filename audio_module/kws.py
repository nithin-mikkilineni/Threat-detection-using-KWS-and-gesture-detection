import sounddevice as sd
import numpy as np
import librosa
import cv2
import tensorflow as tf
from twilio.rest import Client
from collections import deque
import time
import io
from PIL import Image
import matplotlib.cm as cm 
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
VIRIDIS_MAP = cm.get_cmap('viridis')

MODEL_PATH = os.path.join(current_dir, 'my_kws_model_0.h5')
IMG_SIZE = 128
CHANNELS = 3 
WINDOW_SECONDS = 1.0 

SAMPLE_RATE = 22050
N_MELS = 128 
N_FFT = 2048 
HOP_LENGTH = 512 

CATEGORIES = [ 
    'help',
    'unknown', 
    '_background_noise_' 
]

THREAT_CLASSES = ['help' ] 
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
STEP_SECONDS = 0.5
STEP_SAMPLES = int(SAMPLE_RATE * STEP_SECONDS)
BUFFER_SECONDS = 2.0 
BUFFER_SAMPLES = int(BUFFER_SECONDS * SAMPLE_RATE)

CONF_THRESHOLD = 0.90 
SMOOTHING_WINDOW = 1  
detection_history = deque(maxlen=SMOOTHING_WINDOW)
last_alert_time = 0
ALERT_COOLDOWN = 15
try:
    import keys
    ACCOUNT_SID = keys.ACCOUNT_SID
    AUTH_TOKEN = keys.AUTH_TOKEN
    TWILIO_PHONE = keys.TWILIO_PHONE
    YOUR_PHONE = keys.YOUR_PHONE
    SMS_ENABLED = True
    print("[INFO] Keys loaded successfully.")
except ImportError:
    print("[WARNING] 'keys.py' not found in parent directory. SMS disabled.")
except AttributeError:
    print("[WARNING] 'keys.py' missing variables. SMS disabled.")

twilio_client = None
if SMS_ENABLED:
    try:
        twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)
    except Exception as e:
        print(f"[ERROR] Twilio Init Failed: {e}")
        SMS_ENABLED = False
def send_sms(keyword):
    """Sends an SMS alert using Twilio."""
    global sms_sent
    if sms_sent:
        print("SMS already sent for this event.")
        return
    try:
        message = twilio_client.messages.create(
            body=f"EMERGENCY: Threat keyword '{keyword.upper()}' detected!",
            from_=TWILIO_PHONE,
            to=YOUR_PHONE
        )
        print(f"SMS alert sent! SID: {message.sid}")
        sms_sent = True
    except Exception as e:
        print(f"Error sending SMS: {e}")

def audio_to_spectrogram(audio_clip):
    """
    Converts audio clip to a colormapped spectrogram,
    using a fast numpy-based colormap application.
    """
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio_clip,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    
    db_spec = librosa.power_to_db(mel_spec, ref=np.max)
    spec_min = db_spec.min()
    spec_max = db_spec.max()
    if spec_max == spec_min:
        spec_norm = np.zeros_like(db_spec)
    else:
        spec_norm = (db_spec - spec_min) / (spec_max - spec_min)
    
    spec_rgba = VIRIDIS_MAP(spec_norm)

    spec_rgb_float = spec_rgba[:, :, :3]
    spec_rgb_uint8 = (spec_rgb_float * 255).astype(np.uint8)

    resized_spec = cv2.resize(spec_rgb_uint8, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)

    spec_final = resized_spec / 255.0
    
    return spec_final

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

print("\nStarting audio stream... Press Ctrl+C to stop.")
print(f"Listening for threats: {THREAT_CLASSES}")
print("Terminal will remain quiet until a potential threat is detected...")

audio_buffer = np.zeros(BUFFER_SAMPLES, dtype=np.float32)
samples_since_last_step = 0

def audio_callback(indata, frames, t, status): 
    if status:
        print(status)
    global audio_buffer, samples_since_last_step
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata.flatten()
    samples_since_last_step += frames


try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=STEP_SAMPLES,
        callback=audio_callback
    ):
        while True:
            while samples_since_last_step < STEP_SAMPLES:
                time.sleep(0.01)
            
            samples_since_last_step = 0
            
            window = audio_buffer[-WINDOW_SAMPLES:]
            
            spec = audio_to_spectrogram(window)
            
            spec_batch = np.expand_dims(spec, axis=0)
            
            prediction = model.predict(spec_batch, verbose=1)[0]
            
            predicted_index = np.argmax(prediction)
            confidence = prediction[predicted_index]
            predicted_class_name = CATEGORIES[predicted_index]
            
            
            is_threat = False
            if (confidence > CONF_THRESHOLD) and (predicted_class_name in THREAT_CLASSES):
                is_threat = True
             
                print(f"   -> Potential threat: {predicted_class_name} ({confidence*100:.1f}%)")
            
            detection_history.append(is_threat)
            
            if all(detection_history) and time.time() - last_alert_time > ALERT_COOLDOWN:
                
                print(f"\n*** ALERT! Confirmed threat: {predicted_class_name} ***\n")
                if predicted_class_name in THREAT_CLASSES:
                    send_sms(predicted_class_name)
                
                detection_history.clear()
                last_alert_time = time.time()
            
            if time.time() - last_alert_time > ALERT_COOLDOWN:
                sms_sent = False

except KeyboardInterrupt:
    print("\nStopping...")
except Exception as e:
    print(f"An error occurred: {e}")