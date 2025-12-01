# main.py
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
from engagement_tracker import EngagementTracker
from analytics_dashboard import show_dashboard

class EngageSenseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EngageSense AI - Real-Time Student Engagement")
        self.root.geometry("1250x900")
        self.root.configure(bg="#0E1117")

        self.tracker = EngagementTracker()
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.data = []

        self.setup_ui()
        self.process_video()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        title_label = tk.Label(self.root, text="🎓 EngageSense AI",
                               font=("Segoe UI", 22, "bold"), fg="#00BFFF", bg="#0E1117")
        title_label.pack(pady=10)

        self.video_label = ttk.Label(self.root)
        self.video_label.pack(pady=10)

        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=500, mode="determinate")
        self.progress.pack(pady=10)

        self.status_label = tk.Label(self.root, text="Initializing camera...", font=("Segoe UI", 12),
                                     fg="#FFFFFF", bg="#0E1117")
        self.status_label.pack(pady=5)

        self.stop_btn = ttk.Button(self.root, text="🛑 Stop & Show Dashboard", command=self.stop_session)
        self.stop_btn.pack(pady=20)

    def process_video(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.process_video)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.tracker.face_mesh.process(rgb)

        gaze, head, score = "Unknown", "Unknown", 0
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            eye_region = self.tracker.extract_eye_region(landmarks, self.tracker.LEFT_EYE, frame)
            gaze = self.tracker.analyze_gaze(eye_region)
            head = self.tracker.get_head_pose(frame)
            score = self.tracker.compute_engagement(gaze, head)
            eng_state = self.tracker.classify_engagement(score)

            self.data.append({
                "timestamp": time.time() - self.tracker.start_time,
                "gaze": gaze,
                "head": head,
                "engagement": eng_state
            })

            self.progress["value"] = int(score * 100)
            color = "#6bcf7f" if eng_state == "Engaged" else "#ffd93d" if eng_state == "Partially Engaged" else "#ff6b6b"
            self.status_label.config(text=f"🧠 {eng_state} | 👁 Gaze: {gaze} | 💀 Head: {head}", fg=color)

        # Convert frame
        img = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_label.imgtk = img
        self.video_label.configure(image=img)
        self.root.after(10, self.process_video)

    def stop_session(self):
        self.running = False
        self.cap.release()
        show_dashboard(self.root, self.data)

if __name__ == "__main__":
    root = tk.Tk()
    app = EngageSenseApp(root)
    root.mainloop()
