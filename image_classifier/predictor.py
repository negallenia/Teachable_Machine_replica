import numpy as np
import cv2
import json
from tensorflow.keras.models import load_model

# Load model
model = load_model("image_classifier/color_model.h5")

# Load labels
with open("image_classifier/labels.json", "r") as f:
    class_indices = json.load(f)

# Create a reverse mapping from index to label
labels = [label for label, index in sorted(class_indices.items(), key=lambda item: item[1])]

# Image size (should match the model input size)
img_size = (224, 224)  # Change this to match your training image size

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, img_size)
    img_array = np.expand_dims(img / 255.0, axis=0)
    pred = model.predict(img_array)
    label = labels[np.argmax(pred)]

    cv2.putText(frame, f"Prediction: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
