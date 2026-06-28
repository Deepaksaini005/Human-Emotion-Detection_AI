import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model 
print("Loading emotion model...")
model = load_model('emotion_model.h5')
print("Model loaded!")

# Settings 
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

colors = {
    'angry':    (0, 0, 220),
    'disgust':  (0, 140, 255),
    'fear':     (180, 0, 180),
    'happy':    (0, 210, 255),
    'neutral':  (180, 180, 180),
    'sad':      (255, 120, 0),
    'surprise': (0, 255, 160)
}

emoji_text = {
    'angry':    'ANGRY    >:(',
    'disgust':  'DISGUST  eww',
    'fear':     'FEAR     o_o',
    'happy':    'HAPPY    :)',
    'neutral':  'NEUTRAL  :|',
    'sad':      'SAD      :(',
    'surprise': 'SURPRISE :O'
}

# Face detector 
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Webcam 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Webcam started! Press Q to quit.")

# Smooth predictions using rolling average
prediction_history = []
SMOOTH_FRAMES = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)   # mirror the image
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w  = frame.shape[:2]

    # Dark side panel
    panel_width = 220
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_width, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Panel title
    cv2.putText(frame, "EMOTIONS", (w - panel_width + 30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.line(frame, (w - panel_width + 10, 40), (w - 10, 40), (80, 80, 80), 1)

    # Detect faces 
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80)
    )

    current_probs = None

    for (x, y, fw, fh) in faces:
        # Crop + preprocess
        face_roi       = gray[y:y+fh, x:x+fw]
        face_resized   = cv2.resize(face_roi, (48, 48))
        face_norm      = face_resized / 255.0
        face_input     = np.reshape(face_norm, (1, 48, 48, 1))

        # Predict
        raw_pred = model.predict(face_input, verbose=0)[0]

        # Smooth over last N frames
        prediction_history.append(raw_pred)
        if len(prediction_history) > SMOOTH_FRAMES:
            prediction_history.pop(0)
        smoothed = np.mean(prediction_history, axis=0)

        emotion_idx   = np.argmax(smoothed)
        emotion_label = emotions[emotion_idx]
        confidence    = smoothed[emotion_idx] * 100
        color         = colors[emotion_label]
        current_probs = smoothed

        # Face box 
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 2)

        # Rounded label background
        label = f"{emotion_label.upper()}  {confidence:.0f}%"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(frame, (x, y - lh - 20), (x + lw + 16, y), color, -1)
        cv2.putText(frame, label, (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Confidence bar below face box
        bar_y = y + fh + 10
        bar_w = int(fw * (confidence / 100))
        cv2.rectangle(frame, (x, bar_y), (x + fw, bar_y + 8), (60, 60, 60), -1)
        cv2.rectangle(frame, (x, bar_y), (x + bar_w, bar_y + 8), color, -1)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (x, bar_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Side panel bars
    if current_probs is not None:
        for i, (emo, prob) in enumerate(zip(emotions, current_probs)):
            bar_y    = 60 + i * 56
            bar_maxw = panel_width - 30
            bar_fill = int(bar_maxw * prob)
            emo_color = colors[emo]
            is_top   = (i == np.argmax(current_probs))

            # Emotion name
            label_color = (255, 255, 255) if is_top else (160, 160, 160)
            cv2.putText(frame, emoji_text[emo], (w - panel_width + 10, bar_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1)

            # Background bar
            cv2.rectangle(frame,
                          (w - panel_width + 10, bar_y + 6),
                          (w - 10, bar_y + 20),
                          (50, 50, 50), -1)

            # Filled bar
            if bar_fill > 0:
                cv2.rectangle(frame,
                              (w - panel_width + 10, bar_y + 6),
                              (w - panel_width + 10 + bar_fill, bar_y + 20),
                              emo_color, -1)

            # Percentage text
            cv2.putText(frame, f"{prob*100:.0f}%",
                        (w - 38, bar_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, label_color, 1)
    else:
        cv2.putText(frame, "No face", (w - panel_width + 50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)
        cv2.putText(frame, "detected", (w - panel_width + 45, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

    # Bottom bar 
    cv2.rectangle(frame, (0, h - 30), (w - panel_width, h), (20, 20, 20), -1)
    cv2.putText(frame, "Emotion Detector  |  Press Q to quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")