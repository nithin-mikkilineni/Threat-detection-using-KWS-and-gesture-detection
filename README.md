# 🚨 Multimodal Threat Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![Twilio](https://img.shields.io/badge/Twilio-SMS%20API-red?style=for-the-badge&logo=twilio&logoColor=white)

> A modular surveillance system integrating **Audio Keyword Spotting (KWS)** and **Visual Gesture Recognition** to detect distress signals in real-time.

---

## 🧠 System Architecture

This system operates using a central dashboard (`main.py`) that orchestrates two independent deep learning modules. Below is the data flow logic:

```mermaid
graph TD
    A[Start System] -->|Run main.py| B{Select Module}
    
    %% Audio Branch
    B -->|Option 1| C[🎤 Audio Module]
    C --> D[Microphone Input]
    D --> E[Spectrogram Conversion]
    E --> F[CNN Model Inference]
    F -->|Keyword: 'HELP'| G{Confidence > 90%?}

    %% Vision Branch
    B -->|Option 2| H[📷 Vision Module]
    H --> I[Webcam Input]
    I --> J[MediaPipe Hand Tracking]
    J --> K[Keypoint Extraction]
    K --> L[Gesture Model Inference]
    L -->|Gesture: 'HELP'| M{Confidence > 80%?}

    %% Alert Logic
    G -- Yes --> N[🚨 TRIGGER ALERT]
    M -- Yes --> N
    
    N --> O{Cooldown Active?}
    O -- No --> P[Twilio API]
    O -- Yes --> Q[Skip SMS]
    
    P --> R[📲 Send SMS to Emergency Contact]