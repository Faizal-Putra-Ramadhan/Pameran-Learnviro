"""Deteksi emosi realtime (MediaPipe + model sklearn), dijalankan di thread terpisah."""
import collections
import os
import threading
import time
from datetime import datetime

import cv2
import joblib
import mediapipe as mp
import numpy as np

# Terjemahan label folder dataset -> tampilan dashboard.
# Label yang tidak ada di sini ditampilkan apa adanya (huruf pertama kapital).
LABEL_ID = {
    "happy": "Senang", "sad": "Sedih", "angry": "Marah", "neutral": "Netral",
    "surprise": "Terkejut", "surprised": "Terkejut", "fear": "Takut",
    "disgust": "Jijik",
}


def tampilkan(label):
    return LABEL_ID.get(str(label).lower(), str(label).capitalize())


def normalisasi(landmarks):
    """Normalisasi SAMA PERSIS dengan extract_features.py."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    centered = coords - coords[1]
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    if max_dist > 0:
        centered = centered / max_dist
    return centered.flatten()


class EmotionDetector:
    def __init__(self, model_path="emotion_model.pkl", camera_index=0,
                 width=640, height=480):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model '{model_path}' tidak ditemukan.")
        self.model = joblib.load(model_path)
        self.camera_index = camera_index
        self.width, self.height = width, height

        self._lock = threading.Lock()
        self._jpeg = None
        self._emotion = None            # emosi stabil saat ini
        self._votes = collections.deque(maxlen=7)  # smoothing
        self._log = collections.deque(maxlen=20)   # terbaru di depan
        self._counts = collections.Counter()       # distribusi emosi (per frame)
        self._running = False

    # ---- API untuk Flask ----
    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def get_state(self):
        with self._lock:
            return {"emotion": self._emotion, "log": list(self._log),
                    "counts": dict(self._counts)}

    # ---- internal ----
    def _update_emotion(self, label):
        """label = None jika tidak ada wajah. Catat ke log saat emosi berubah."""
        self._votes.append(label)
        stable = collections.Counter(self._votes).most_common(1)[0][0]
        with self._lock:
            if stable is not None:
                self._counts[stable] += 1
            if stable != self._emotion:
                self._emotion = stable
                if stable is not None:
                    self._log.appendleft({
                        "waktu": datetime.now().strftime("%H:%M"),
                        "hasil": stable,
                    })

    def _open_camera(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def _loop(self):
        mp_fm = mp.solutions.face_mesh
        drawing = mp.solutions.drawing_utils
        styles = mp.solutions.drawing_styles
        face_mesh = mp_fm.FaceMesh(max_num_faces=5, refine_landmarks=False,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)
        cap = self._open_camera()
        gagal = 0

        while self._running:
            ok, image = cap.read()
            if not ok:
                gagal += 1
                if gagal > 30:           # kamera putus -> buka ulang
                    cap.release()
                    time.sleep(1)
                    cap = self._open_camera()
                    gagal = 0
                time.sleep(0.05)
                continue
            gagal = 0

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)

            labels = []
            if results.multi_face_landmarks:
                feats = [normalisasi(f.landmark) for f in results.multi_face_landmarks]
                preds = self.model.predict(np.array(feats))
                h, w, _ = image.shape
                for face, pred in zip(results.multi_face_landmarks, preds):
                    teks = tampilkan(pred)
                    labels.append(teks)
                    drawing.draw_landmarks(
                        image=image, landmark_list=face,
                        connections=mp_fm.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=styles.get_default_face_mesh_tesselation_style())
                    x = max(10, min(int(face.landmark[10].x * w) - 40, w - 150))
                    y = max(30, min(int(face.landmark[10].y * h) - 15, h - 10))
                    cv2.putText(image, teks, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Jika banyak wajah, ambil emosi yang paling banyak muncul
            self._update_emotion(
                collections.Counter(labels).most_common(1)[0][0] if labels else None)

            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self._lock:
                    self._jpeg = buf.tobytes()
