import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load model and label classes
model = tf.keras.models.load_model("pose_model.keras")
label_classes = np.load("label_classes.npy")

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Extract keypoints from MediaPipe result
def extract_keypoints(results):
    if not results.pose_landmarks:
        return np.zeros(33 * 4)
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(keypoints)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural webcam feel
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    # Draw landmarks
    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Predict if pose detected
    keypoints = extract_keypoints(results)
    if keypoints.sum() != 0:
        input_data = keypoints.reshape(1, -1)
        prediction = model.predict(input_data, verbose=0)
        pred_label = label_classes[np.argmax(prediction)]
        conf = np.max(prediction)

        # Show prediction on frame
        cv2.putText(frame, f"{pred_label} ({conf*100:.1f}%)",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 3)

    cv2.imshow("Pose Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
