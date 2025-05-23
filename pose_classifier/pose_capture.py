import cv2
import os
import numpy as np
import mediapipe as mp

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

def extract_keypoints(results):
    if not results.pose_landmarks:
        return [0] * 33 * 4
    return np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark]).flatten().tolist()

def collect_pose_data(label, num_samples=100, save_dir='pose_data'):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    collecting = False

    print(f"📸 Ready to collect poses for label '{label}'")
    print("Press SPACE to start, Q to quit")

    # Define drawing lines
    landmark_drawing_spec = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4)
    connection_drawing_spec = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=4)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("FAILED to capture")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        # Draw pose with custom lines
        mp_draw.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_drawing_spec,
            connection_drawing_spec=connection_drawing_spec
        )

        if collecting and count < num_samples:
            keypoints = extract_keypoints(results)
            if sum(keypoints) != 0:
                save_path = os.path.join(save_dir, f"{label}_{count}.csv")
                with open(save_path, 'w') as f:
                    f.write(','.join(map(str, keypoints)))
                print(f"✅ {save_path}")
                count += 1

        # Status message
        status_text = f"{'Collecting' if collecting else 'Waiting'}: {label} ({count}/{num_samples})"
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if collecting else (0, 255, 255), 2)

        cv2.imshow("Pose Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # SPACEBAR
            collecting = not collecting
            print("▶️ Started collecting" if collecting else "⏸️ Paused collecting")

        if count >= num_samples:
            print("✅ Reached desired sample count.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("📁 Done.")

# Example usage
if __name__ == "__main__":
    collect_pose_data("thinker", num_samples=50)
