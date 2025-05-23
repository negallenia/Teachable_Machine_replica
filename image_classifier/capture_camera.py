import cv2
import os

def capture_images(label, output_dir='data', num_samples=100):
    cap = cv2.VideoCapture(0)
    save_path = os.path.join(output_dir, label)
    os.makedirs(save_path, exist_ok=True)

    count = 0
    print(f"📸 Capturing {num_samples} images for label: '{label}'")
    print("Press SPACE to capture an image")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Show progress on the image
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Label: {label} | Captured: {count}/{num_samples}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Capture - Press SPACE to Save, Q to Quit", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == 32:  # SPACE key
            img_path = os.path.join(save_path, f"{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"✅ Saved {img_path}")
            count += 1

            if count >= num_samples:
                print("✅ Image capture complete.")
                break

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    capture_images("red")  # Replace "wave" with your label (e.g., "jump", "sit")
