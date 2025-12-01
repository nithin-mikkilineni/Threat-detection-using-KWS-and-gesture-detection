import cv2
import mediapipe as mp  
import numpy as np
import tensorflow as tf
import os
import time
from twilio.rest import Client
import sys

# 1. Project Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..')))
MODEL_PATH = os.path.join(CURRENT_DIR, "gesture_model_multiclass.h5")
LABELS_PATH = os.path.join(CURRENT_DIR, "class_labels.txt")
# 2. Twilio Credentials
try:
    import keys
    ACCOUNT_SID = keys.ACCOUNT_SID
    AUTH_TOKEN = keys.AUTH_TOKEN
    TWILIO_PHONE = keys.TWILIO_PHONE
    YOUR_PHONE = keys.YOUR_PHONE
    SMS_ENABLED = True
except ImportError:
    print("[WARNING] 'keys.py' not found. SMS alerts will be DISABLED.")
    SMS_ENABLED = False
    ACCOUNT_SID = None
    AUTH_TOKEN = None
# 3. Alert Settings
CONFIDENCE_THRESHOLD = 0.8
SMS_COOLDOWN_SECONDS = 30 


class PatchedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config.pop('groups', None)
        return super().from_config(config)

class GestureRecognizer:
    def __init__(self, model_path, labels_path, confidence_threshold=0.75):
       
        tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = PatchedDepthwiseConv2D
        
        print(f"[INFO] Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        self.conf_threshold = confidence_threshold

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        with open(labels_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        print(f"[INFO] Classes loaded: {self.class_names}")

    def preprocess_hand(self, hand_img):
        """Apply normalization and resize for model input."""
        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(hand_img, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        hand_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        hand_img = cv2.resize(hand_img, (224, 224))
        hand_img = hand_img.astype("float32") / 255.0
        hand_img = np.expand_dims(hand_img, axis=0)
        return hand_img

    def detect_gesture(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None  

        hand_landmarks = results.multi_hand_landmarks[0]
        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        x1, y1 = int(min(x_coords) * w), int(min(y_coords) * h)
        x2, y2 = int(max(x_coords) * w), int(max(y_coords) * h)

        pad = int(0.2 * (x2 - x1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        hand_img = frame[y1:y2, x1:x2]
        if hand_img.size == 0:
            return None 

        hand_img = self.preprocess_hand(hand_img)

        preds = self.model.predict(hand_img, verbose=0)
        gesture_idx = np.argmax(preds)
        confidence = float(np.max(preds))
        gesture_name = self.class_names[gesture_idx]

        if confidence < self.conf_threshold:
            gesture_name = "Unknown"

        color = (0, 255, 0) if gesture_name != "Unknown" else (0, 0, 255)
        
        # Draw UI
        cv2.putText(frame, f"{gesture_name.upper()} ({confidence:.2f})",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if gesture_name != "Unknown":
            return gesture_name
        else:
            return None

def send_twilio_alert(client, gesture_name):
    """Sends an SMS via Twilio."""
    try:
        message = client.messages.create(
            body=f"⚠️ THREAT DETECTED: {gesture_name} recognized by surveillance system.",
            from_=TWILIO_PHONE,
            to=YOUR_PHONE
        )
        print(f"[SMS SENT] SID: {message.sid}")
    except Exception as e:
        print(f"[SMS ERROR] Could not send message: {e}")

def main():
    # 1. Initialize Twilio Client
    sms_client = None
    if SMS_ENABLED:
        try:
            sms_client = Client(ACCOUNT_SID, AUTH_TOKEN)
            print("[INFO] Twilio Client Initialized.")
        except Exception as e:
            print(f"[WARNING] Twilio init failed: {e}. SMS will not be sent.")
    else:
        print("[INFO] Running in Offline Mode (No SMS).")

    # 2. Initialize Gesture Recognizer
    try:
        gesture_recognizer = GestureRecognizer(MODEL_PATH, LABELS_PATH, CONFIDENCE_THRESHOLD)
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return

    # 3. Start Camera
    cap = cv2.VideoCapture(0)
    print("[INFO] Starting real-time gesture detection. Press 'Q' to quit.")

    last_sms_time = 0 # Timestamp of the last sent SMS

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for mirror effect (optional, easier for gestures)
        frame = cv2.flip(frame, 1)

        # Detect Gesture
        detected_gesture = gesture_recognizer.detect_gesture(frame)

        # Check for Alert Condition
        if detected_gesture and detected_gesture.lower() == "help":
            current_time = time.time()
            
            # Check if enough time has passed since the last SMS (Cooldown)
            if (current_time - last_sms_time) > SMS_COOLDOWN_SECONDS:
                print(f"[ALERT] Threat Detected: {detected_gesture}")
                
                if sms_client:
                    send_twilio_alert(sms_client, detected_gesture)
                
                last_sms_time = current_time # Reset timer
            else:
                # Optional: Visual indicator that SMS is on cooldown
                cv2.putText(frame, "SMS Cooldown Active", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Gesture Threat Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()