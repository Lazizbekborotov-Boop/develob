from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os
import threading
import math

app = Flask(__name__)

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

FINGER_TIPS = [4, 8, 12, 16, 20]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

hand_data = {
    "index_tip": {"x": 0.5, "y": 0.5},
    "pinch": False,
    "pinch_dist": 0.1,
    "fingers": [0,0,0,0,0],
    "hands": 0,
    "landmarks": []
}
lock = threading.Lock()

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Model yuklanmoqda...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Tayyor!")

def count_fingers(landmarks, handedness):
    fingers = []
    lm = landmarks
    if handedness == "Right":
        fingers.append(1 if lm[4].x < lm[3].x else 0)
    else:
        fingers.append(1 if lm[4].x > lm[3].x else 0)
    for tip in FINGER_TIPS[1:]:
        fingers.append(1 if lm[tip].y < lm[tip - 2].y else 0)
    return fingers

def get_pinch_dist(lm):
    t = np.array([lm[4].x, lm[4].y])
    i = np.array([lm[8].x, lm[8].y])
    return float(np.linalg.norm(t - i))

def draw_neon_hand(frame, landmarks, color=(0, 255, 220)):
    h, w, _ = frame.shape
    pts = {}
    for i, lm in enumerate(landmarks):
        pts[i] = (int(lm.x * w), int(lm.y * h))

    # Glow effect: draw multiple times with decreasing thickness
    for thickness, alpha in [(8, 0.15), (5, 0.25), (3, 0.5), (2, 1.0)]:
        overlay = frame.copy()
        for conn in HAND_CONNECTIONS:
            cv2.line(overlay, pts[conn[0]], pts[conn[1]], color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha if alpha < 1 else 1, frame, 1 - (alpha if alpha < 1 else 0), 0, frame)

    for i in range(21):
        r = 8 if i in FINGER_TIPS else 5
        cv2.circle(frame, pts[i], r + 3, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pts[i], r, (255, 255, 255), -1, cv2.LINE_AA)

def generate_frames():
    download_model()
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Dark overlay on camera
        dark = np.zeros_like(frame)
        cv2.addWeighted(frame, 0.25, dark, 0.75, 0, frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        index_tip = {"x": 0.5, "y": 0.5}
        pinch = False
        pinch_dist = 0.1
        fingers = [0,0,0,0,0]
        lm_list = []

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            handedness = result.handedness[0][0].display_name
            fingers = count_fingers(lm, handedness)
            pinch_dist = get_pinch_dist(lm)
            pinch = pinch_dist < 0.065

            index_tip = {"x": float(lm[8].x), "y": float(lm[8].y)}
            lm_list = [{"x": float(p.x), "y": float(p.y)} for p in lm]

            draw_neon_hand(frame, lm)

        with lock:
            hand_data["index_tip"] = index_tip
            hand_data["pinch"] = pinch
            hand_data["pinch_dist"] = float(pinch_dist)
            hand_data["fingers"] = fingers
            hand_data["hands"] = len(result.hand_landmarks) if result.hand_landmarks else 0
            hand_data["landmarks"] = lm_list

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hand_data')
def get_hand_data():
    with lock:
        return jsonify(hand_data)

if __name__ == '__main__':
    print("\n🖐  Neon Hand Control — http://localhost:5000\n")
    app.run(debug=False, threaded=True)