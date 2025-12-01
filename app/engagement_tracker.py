import cv2
import dlib
import mediapipe as mp
import numpy as np
import os
import time
from collections import deque


class EngagementTracker:
    def __init__(self):
        # Mediapipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_faces=1
        )

        # Eye indices
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173]

        # Head pose via dlib
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = "models/shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(predictor_path) if os.path.exists(predictor_path) else None

        # Rolling history
        self.gaze_history = deque(maxlen=60)
        self.head_history = deque(maxlen=60)
        self.start_time = time.time()
        self.analysis_data = []

    # ----------- Eye tracking -----------
    def extract_eye_region(self, landmarks, eye_indices, frame):
        points = [(int(landmarks.landmark[idx].x * frame.shape[1]),
                   int(landmarks.landmark[idx].y * frame.shape[0])) for idx in eye_indices]
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

        # Convert to grayscale
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)

        # --- STEP 1: Detect overexposure and compensate ---
        mean_intensity = np.mean(gray)
        if mean_intensity > 130:
            # If the region is too bright, darken it dynamically
            factor = 130.0 / mean_intensity
            gray = cv2.convertScaleAbs(gray, alpha=factor, beta=-40)
        elif mean_intensity < 60:
            # If too dark, brighten slightly
            gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=20)

        # --- STEP 2: Reduce glare reflections ---
        gray = cv2.medianBlur(gray, 5)
        gray = cv2.bilateralFilter(gray, 5, 50, 50)

        # --- STEP 3: Local contrast enhancement (CLAHE) ---
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # --- STEP 4: Adaptive thresholding (dynamic per lighting) ---
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # --- STEP 5: Contour detection for pupil localization ---
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "Unknown"
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return "Unknown"

        cx = int(M["m10"] / M["m00"])
        norm_x = cx / eye_region.shape[1]

        # --- STEP 6: Final classification ---
        if norm_x < 0.35:
            return "Left"
        elif norm_x > 0.65:
            return "Right"
        else:
            return "Center"

    # ----------- Head pose -----------
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

    # ----------- Engagement Logic -----------
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
