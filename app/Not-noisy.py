# pyqt6_app_stable.py
# EngageSense AI — Stable Version (with Temporal Smoothing)
# Reduces noisy fluctuations in engagement predictions

from __future__ import annotations
import sys, time, os, statistics
from collections import deque
import cv2, numpy as np, mediapipe as mp, dlib
import pandas as pd
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QProgressBar
from PyQt6.QtGui import QPixmap, QImage


# -----------------------------
# Engagement Tracker
# -----------------------------
class EngagementTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_faces=1
        )
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173]

        self.detector = dlib.get_frontal_face_detector()
        predictor_path = "models/shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(predictor_path) if os.path.exists(predictor_path) else None
        self.start_time = time.time()

    def extract_eye_region(self, landmarks, eye_indices, frame):
        points = [(int(landmarks.landmark[i].x * frame.shape[1]),
                   int(landmarks.landmark[i].y * frame.shape[0])) for i in eye_indices]
        x_coords, y_coords = [p[0] for p in points], [p[1] for p in points]
        x_min, x_max = max(0, min(x_coords) - 5), min(frame.shape[1], max(x_coords) + 5)
        y_min, y_max = max(0, min(y_coords) - 5), min(frame.shape[0], max(y_coords) + 5)
        eye_region = frame[y_min:y_max, x_min:x_max]
        if eye_region.size > 0:
            eye_region = cv2.resize(eye_region, (100, 50))
        return eye_region

    def analyze_gaze(self, eye_region):
        if eye_region is None or eye_region.size == 0:
            return "Unknown"
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "Unknown"
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return "Unknown"
        cx = int(M["m10"] / M["m00"])
        norm_x = cx / eye_region.shape[1]
        if norm_x < 0.35:
            return "Left"
        elif norm_x > 0.65:
            return "Right"
        else:
            return "Center"

    def get_head_pose(self, frame):
        if self.predictor is None:
            return "Unknown"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) == 0:
            return "Unknown"
        shape = self.predictor(gray, faces[0])
        image_points = np.array([
            (shape.part(30).x, shape.part(30).y),
            (shape.part(8).x, shape.part(8).y),
            (shape.part(36).x, shape.part(36).y),
            (shape.part(45).x, shape.part(45).y),
            (shape.part(48).x, shape.part(48).y),
            (shape.part(54).x, shape.part(54).y)
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1)
        ])
        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]])
        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        if not success:
            return "Unknown"
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        yaw = angles[1]
        if yaw < -15:
            return "Left"
        elif yaw > 15:
            return "Right"
        else:
            return "Center"

    def compute_engagement(self, gaze, head):
        gaze_val = 1 if gaze == "Center" else 0.5 if gaze == "Unknown" else 0
        head_val = 1 if head == "Center" else 0.5 if head == "Unknown" else 0
        score = 0.6 * gaze_val + 0.4 * head_val
        return score

    def classify_engagement(self, score):
        if score >= 0.7:
            return "Engaged"
        elif score >= 0.4:
            return "Partially Engaged"
        else:
            return "Not Engaged"


# -----------------------------
# Main Application (Stable)
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎓 EngageSense AI — Stable Detection")
        self.resize(1250, 900)

        # --- Layout ---
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.video_lbl = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.video_lbl.setMinimumSize(960, 540)
        layout.addWidget(self.video_lbl)

        self.status_lbl = QLabel("Ready...", alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_lbl)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶️ Start")
        self.stop_btn = QPushButton("🛑 Stop")
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        # --- Logic ---
        self.cap = None
        self.tracker = EngagementTracker()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._process_frame)
        self.running = False
        self.data = []

        # smoothing memory
        self.score_history = deque(maxlen=15)
        self.state_history = deque(maxlen=15)
        self.last_state = None
        self.last_change_time = time.time()

        # connections
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_lbl.setText("❌ Could not open camera.")
            return
        self.running = True
        self.timer.start(15)
        self.status_lbl.setText("Recording...")

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.data:
            pd.DataFrame(self.data).to_csv("stable_predictions.csv", index=False)
        self.status_lbl.setText("Stopped and saved ✅")

    @QtCore.pyqtSlot()
    def _process_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.tracker.face_mesh.process(rgb)

        gaze, head, score, state = "Unknown", "Unknown", 0.0, "Unknown"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            eye_region = self.tracker.extract_eye_region(landmarks, self.tracker.LEFT_EYE, frame)
            gaze = self.tracker.analyze_gaze(eye_region)
            head = self.tracker.get_head_pose(frame)
            score = self.tracker.compute_engagement(gaze, head)

            # smooth score
            self.score_history.append(score)
            avg_score = statistics.mean(self.score_history)

            state = self.tracker.classify_engagement(avg_score)
            self.state_history.append(state)

            # majority vote over recent frames
            stable_state = max(set(self.state_history), key=self.state_history.count)

            # debounce changes
            now = time.time()
            if stable_state != self.last_state and (now - self.last_change_time) < 1.0:
                stable_state = self.last_state
            else:
                self.last_state = stable_state
                self.last_change_time = now

            color = "#6bcf7f" if stable_state == "Engaged" else "#ffd93d" if stable_state == "Partially Engaged" else "#ff6b6b"
            self.status_lbl.setText(f"🧠 {stable_state} | 👁 {gaze} | 💀 {head}")
            self.progress.setValue(int(avg_score * 100))
            self.progress.setStyleSheet(f"QProgressBar::chunk {{background-color:{color};}}")

            self.data.append({
                "timestamp": time.time() - self.tracker.start_time,
                "gaze": gaze,
                "head": head,
                "score": round(avg_score, 3),
                "engagement": stable_state
            })

        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_lbl.setPixmap(QPixmap.fromImage(qimg))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
