# pyQt6_app.py
# ============================================
# EngageSense AI – Live Engagement + Analytics
# No CSV dataset, only live monitoring + report
# ============================================

from __future__ import annotations
import sys
import time
import os

from typing import List, Dict, Any

import cv2
import numpy as np
import mediapipe as mp
import dlib

from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QVBoxLayout, QHBoxLayout, QProgressBar
)
from PyQt6.QtGui import QPixmap, QImage

import pandas as pd

# -----------------------------
# Engagement Tracker
# -----------------------------
class EngagementTracker:
    def __init__(self):
        # MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_faces=1
        )

        # Eye indices
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173]

        # Head pose (dlib)
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = "models/shape_predictor_68_face_landmarks.dat"
        if os.path.exists(predictor_path):
            self.predictor = dlib.shape_predictor(predictor_path)
        else:
            print("⚠️ Warning: shape_predictor_68_face_landmarks.dat not found!")
            self.predictor = None

        # Session timing + data collection
        self.start_time = time.time()
        self.data: List[Dict[str, Any]] = []

    # ---------------- Eye Region Extraction ----------------
    def extract_eye_region(self, landmarks, eye_indices, frame):
        h, w, _ = frame.shape
        points = [
            (int(landmarks.landmark[i].x * w),
             int(landmarks.landmark[i].y * h))
            for i in eye_indices
        ]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = max(0, min(xs) - 5), min(w, max(xs) + 5)
        y_min, y_max = max(0, min(ys) - 5), min(h, max(ys) + 5)
        region = frame[y_min:y_max, x_min:x_max]
        if region.size > 0:
            region = cv2.resize(region, (100, 50))
        return region

    # ---------------- Gaze Analysis ----------------
    def analyze_gaze(self, eye_region):
        if eye_region is None or eye_region.size == 0:
            return "Unknown"

        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
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

    # ---------------- Head Pose ----------------
    def get_head_pose(self, frame):
        if self.predictor is None:
            return "Unknown"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) == 0:
            return "Unknown"

        shape = self.predictor(gray, faces[0])

        image_points = np.array([
            (shape.part(30).x, shape.part(30).y),  # Nose tip
            (shape.part(8).x, shape.part(8).y),    # Chin
            (shape.part(36).x, shape.part(36).y),  # Left eye left corner
            (shape.part(45).x, shape.part(45).y),  # Right eye right corner
            (shape.part(48).x, shape.part(48).y),  # Left mouth corner
            (shape.part(54).x, shape.part(54).y)   # Right mouth corner
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

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ])
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )
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

    # ---------------- Engagement Logic ----------------
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
# Main Window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎓 EngageSense AI — Live Engagement Monitoring")
        self.resize(1250, 900)

        # Central layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Video label
        self.video_lbl = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.video_lbl.setMinimumSize(960, 540)
        layout.addWidget(self.video_lbl)

        # Status label
        self.status_lbl = QLabel("Ready...", alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_lbl)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        # Buttons
        self.start_btn = QPushButton("▶️ Start Monitoring")
        self.stop_btn = QPushButton("🛑 Stop & Show Report")
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        # Signals
        self.start_btn.clicked.connect(self.start_capture)
        self.stop_btn.clicked.connect(self.stop_capture)

        # Core stuff
        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._process_frame)
        self.tracker = EngagementTracker()
        self.running = False

    # ---------------- Start Capture ----------------
    def start_capture(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_lbl.setText("❌ Could not open camera.")
            return

        self.running = True
        self.tracker.start_time = time.time()
        self.tracker.data.clear()

        self.timer.start(15)
        self.status_lbl.setText("Monitoring...")

    # ---------------- Stop Capture ----------------
    def stop_capture(self):
        if not self.running:
            return

        self.running = False
        self.timer.stop()

        if self.cap:
            self.cap.release()
            self.cap = None

        self.status_lbl.setText("Session ended. Opening analytics dashboard...")

        # Show dashboard (Tkinter window)
        if self.tracker.data:
            from dashboard_window import show_dashboard
            show_dashboard(self.tracker.data)
        else:
            self.status_lbl.setText("No data collected. Try again.")

    # ---------------- Frame Processing ----------------
    @QtCore.pyqtSlot()
    def _process_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.tracker.face_mesh.process(rgb)

        gaze, head = "Unknown", "Unknown"
        score, eng_state = 0.0, "Unknown"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            eye_region = self.tracker.extract_eye_region(
                landmarks, self.tracker.LEFT_EYE, frame
            )
            gaze = self.tracker.analyze_gaze(eye_region)
            head = self.tracker.get_head_pose(frame)
            score = self.tracker.compute_engagement(gaze, head)
            eng_state = self.tracker.classify_engagement(score)

        # Save to session data
        self.tracker.data.append({
            "timestamp": time.time() - self.tracker.start_time,
            "gaze": gaze,
            "head": head,
            "score": float(score),
            "engagement": eng_state
        })

        # Update UI
        self.progress.setValue(int(score * 100))
        color = (
            "#6bcf7f" if eng_state == "Engaged"
            else "#ffd93d" if eng_state == "Partially Engaged"
            else "#ff6b6b"
        )
        self.progress.setStyleSheet(f"QProgressBar::chunk {{background-color:{color};}}")
        self.status_lbl.setText(f"🧠 {eng_state} | 👁 {gaze} | 💀 {head}")

        # Display frame
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_lbl.setPixmap(QPixmap.fromImage(qimg))


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
