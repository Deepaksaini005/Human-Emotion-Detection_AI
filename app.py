import os
from collections import deque

import cv2
import numpy as np
from tensorflow.keras.models import load_model

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'emotion_model.h5')
SMOOTH_FRAMES = 5

COLORS = {
    'angry': (0, 0, 220),
    'disgust': (0, 140, 255),
    'fear': (180, 0, 180),
    'happy': (0, 210, 255),
    'neutral': (180, 180, 180),
    'sad': (255, 120, 0),
    'surprise': (0, 255, 160)
}

EMOJI_TEXT = {
    'angry': 'ANGRY    >:(',
    'disgust': 'DISGUST  eww',
    'fear': 'FEAR     o_o',
    'happy': 'HAPPY    :)',
    'neutral': 'NEUTRAL  :|',
    'sad': 'SAD      :(',
    'surprise': 'SURPRISE :O'
}


def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Unable to find model file at {MODEL_PATH}")

    print('Loading emotion model...')
    model = load_model(MODEL_PATH)
    print('Model loaded!')
    return model


def preprocess_face_image(face_image):
    if face_image is None:
        raise ValueError('face_image cannot be None')

    gray = face_image
    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    gray = gray.astype('float32') / 255.0
    return gray.reshape(1, 48, 48, 1)


def open_camera(preferred_width=640, preferred_height=480, max_index=4):
    backends = []
    try:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    except Exception:
        backends = [cv2.CAP_ANY]

    for backend in backends:
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx, backend)
            if not cap or not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, preferred_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preferred_height)
            print(f'Opened camera index {idx} with backend {backend}. Press Q to quit.')
            return cap

    fallback = 'test_video.mp4'
    try:
        cap = cv2.VideoCapture(fallback)
        if cap and cap.isOpened():
            print(f'No webcam found — opened fallback video {fallback}. Press Q to quit.')
            return cap
        try:
            cap.release()
        except Exception:
            pass
    except Exception:
        pass

    print('ERROR: Could not open any webcam or fallback video.\n'
          '- If you are on Windows, try connecting a camera and ensure drivers are installed.\n'
          '- To force a specific camera, set the index in the code or pass a video file path.')
    return None


def run_detection():
    model = load_emotion_model()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise RuntimeError('Face detector model could not be loaded.')

    cap = open_camera()
    if cap is None:
        raise SystemExit(1)

    print('Webcam started! Press Q to quit.')
    prediction_history = deque(maxlen=SMOOTH_FRAMES)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            h, w = frame.shape[:2]

            panel_width = 220
            overlay = frame.copy()
            cv2.rectangle(overlay, (w - panel_width, 0), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            cv2.putText(frame, 'EMOTIONS', (w - panel_width + 30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.line(frame, (w - panel_width + 10, 40), (w - 10, 40), (80, 80, 80), 1)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70)
            )
            faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)

            current_probs = None
            if faces:
                x, y, fw, fh = faces[0]
                pad_x = int(fw * 0.2)
                pad_y = int(fh * 0.2)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + fw + pad_x)
                y2 = min(h, y + fh + pad_y)

                face_roi = gray[y1:y2, x1:x2]
                if face_roi.size > 0:
                    face_input = preprocess_face_image(face_roi)
                    raw_pred = model.predict(face_input, verbose=0)[0]
                    prediction_history.append(raw_pred)
                    smoothed = np.mean(prediction_history, axis=0)

                    emotion_idx = int(np.argmax(smoothed))
                    emotion_label = EMOTIONS[emotion_idx]
                    confidence = float(smoothed[emotion_idx] * 100.0)
                    color = COLORS[emotion_label]
                    current_probs = smoothed

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{emotion_label.upper()}  {confidence:.0f}%'
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                    cv2.rectangle(frame, (x1, y1 - lh - 20), (x1 + lw + 16, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 8, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                    bar_y = y2 + 10
                    bar_w = int((x2 - x1) * (confidence / 100.0))
                    cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + 8), (60, 60, 60), -1)
                    cv2.rectangle(frame, (x1, bar_y), (x1 + bar_w, bar_y + 8), color, -1)
                    cv2.putText(frame, f'Confidence: {confidence:.1f}%', (x1, bar_y + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            if current_probs is not None:
                for i, (emo, prob) in enumerate(zip(EMOTIONS, current_probs)):
                    bar_y = 60 + i * 56
                    bar_maxw = panel_width - 30
                    bar_fill = int(bar_maxw * prob)
                    emo_color = COLORS[emo]
                    is_top = (i == int(np.argmax(current_probs)))
                    label_color = (255, 255, 255) if is_top else (160, 160, 160)
                    cv2.putText(frame, EMOJI_TEXT[emo], (w - panel_width + 10, bar_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1)
                    cv2.rectangle(frame, (w - panel_width + 10, bar_y + 6),
                                  (w - 10, bar_y + 20), (50, 50, 50), -1)
                    if bar_fill > 0:
                        cv2.rectangle(frame, (w - panel_width + 10, bar_y + 6),
                                      (w - panel_width + 10 + bar_fill, bar_y + 20), emo_color, -1)
                    cv2.putText(frame, f'{prob * 100:.0f}%', (w - 38, bar_y + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, label_color, 1)
            else:
                cv2.putText(frame, 'No face', (w - panel_width + 50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)
                cv2.putText(frame, 'detected', (w - panel_width + 45, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

            cv2.rectangle(frame, (0, h - 30), (w - panel_width, h), (20, 20, 20), -1)
            cv2.putText(frame, 'Emotion Detector  |  Press Q to quit',
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.imshow('Emotion Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print('Webcam closed.')


if __name__ == '__main__':
    run_detection()