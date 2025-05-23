import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import mediapipe as mp
from tensorflow.keras.models import load_model
import json

# Load image labels
def load_image_labels(path="image_classifier/labels.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            label_dict = json.load(f)
            # Invert the dictionary: index (as str) -> name
            return {str(v): k for k, v in label_dict.items()}
    return {}

# Load pose labels
def load_pose_labels(path="pose_classifier/label_classes.npy"):
    if os.path.exists(path):
        return np.load(path)
    return []

# ----------------- Pose Utilities -----------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_detector = mp_pose.Pose()


def extract_keypoints(results):
    if results.pose_landmarks:
        return np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark]).flatten()
    return np.zeros(132)


def load_pose_model(model_path="pose_classifier/pose_model.keras"):
    if os.path.exists(model_path):
        return load_model(model_path)
    return None


def predict_pose(frame, model, label_map):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(rgb)
    keypoints = extract_keypoints(results)
    if np.sum(keypoints) == 0:
        return None, frame
    if model:
        probs = model.predict(np.array([keypoints]), verbose=0)[0]
        if label_map is not None and len(label_map) > 0:
            prediction = label_map[np.argmax(probs)]
    else:
        prediction = np.argmax(probs)
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return prediction, frame


# ----------------- Image Utilities -----------------
def load_image_model(model_path="image_classifier/color_model.h5"):
    if os.path.exists(model_path):
        return load_model(model_path)
    return None


def predict_image(frame, model, label_map):
    resized = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    if model:
        probs = model.predict(img_array, verbose=0)[0]
        idx = int(np.argmax(probs))
        if isinstance(label_map, dict):
            prediction = label_map.get(str(idx), str(idx))
        elif isinstance(label_map, list) and len(label_map) > idx:
            prediction = label_map[idx]
        else:
            prediction = idx
    else:
        prediction = "No model"
    return prediction, frame


# ----------------- GUI -----------------
class ClassifierGUI:
    def __init__(self, root):
        self.image_labels = load_image_labels()
        self.pose_labels = load_pose_labels()
        self.root = root
        self.root.title("Pose & Image Classifier")

        self.video_label = ttk.Label(root)
        self.video_label.pack()

        self.model_type = tk.StringVar(value="pose")
        ttk.Radiobutton(root, text="Pose Classifier", variable=self.model_type, value="pose").pack(side=tk.LEFT)
        ttk.Radiobutton(root, text="Image Classifier", variable=self.model_type, value="image").pack(side=tk.LEFT)

        self.prediction_label = ttk.Label(root, text="Prediction: ...", font=("Arial", 16))
        self.prediction_label.pack(pady=10)

        self.cap = cv2.VideoCapture(0)

        self.pose_model = load_pose_model()
        self.image_model = load_image_model()

        self.running = True
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        if self.model_type.get() == 'pose':
            prediction, processed = predict_pose(frame.copy(), self.pose_model, self.pose_labels)
        else:
            prediction, processed = predict_image(frame.copy(), self.image_model, self.image_labels)

        self.prediction_label.config(text=f"Prediction: {prediction}")

        img = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        if self.running:
            self.root.after(10, self.update_video)

    def on_closing(self):
        self.running = False
        self.cap.release()
        self.root.destroy()


# ----------------- Main -----------------
if __name__ == '__main__':
    root = tk.Tk()
    app = ClassifierGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
