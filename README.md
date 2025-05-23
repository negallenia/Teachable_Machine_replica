# Teachable_Machine_replica

This project is a **Teachable Machine-style Lab Assistant** that helps users create, train, and test machine learning models using webcam input. It can classify both **human body poses** and **captured images**, using a range of machine learning and deep learning models. The system works entirely offline, and includes:

---

1. **Data Collection (with webcam)**

* Users can collect image or pose data directly using the webcam.
* Pose data is collected using **MediaPipe**, extracting 33 human body landmarks per frame.
* Image data is saved as raw webcam images for training.

---

2. **Labeling & Preprocessing**

* **Image labels** are stored in a `labels.json` file.
* **Pose labels** (keypoint sequences) are saved in `.npy` format for fast loading.
* Data is automatically flattened and normalized before feeding into a model.

---

3. **Training Options**

You can train different types of models on the collected data:

  A. Classical Machine Learning

* **CatBoostClassifier** or **RandomForestClassifier** for pose or image classification.
* Good for small datasets, fast training, interpretable models.
* Pose model uses extracted keypoints, image model uses flattened, resized RGB arrays.

  B. Regression Example

* A regression model can be trained (e.g., predicting an angle or distance from pose keypoints).
* Useful for tracking metrics like limb length, orientation, or movement intensity.

  C. Deep Learning – Neural Networks (Keras)

* For image classification: a **Convolutional Neural Network (CNN)** using Keras.
* For pose classification: a **simple Dense Neural Network** that learns from pose keypoint arrays.
* Models are saved as `.h5` or `.keras` and reloaded at runtime.

---

 4. **Live Prediction Interface (GUI)**

* The app uses **Tkinter** to display the webcam feed and prediction label in real time.
* Users can switch between **pose classifier** and **image classifier** modes using radio buttons.
* The system automatically loads the appropriate model and label map on startup.

---

5. **How It Works Internally**

  A. Pose Classifier

1. MediaPipe extracts 132D pose keypoints from the webcam feed.
2. Keypoints are passed into a trained model (e.g., RandomForest or Keras DNN).
3. The predicted label is displayed on screen with the body landmarks drawn.

  B. Image Classifier

1. Webcam frames are resized (e.g., 64×64), flattened, and normalized.
2. The image is passed to a trained classifier (e.g., CNN or CatBoost).
3. The predicted class (e.g., red object, green object) is displayed on screen.

---

Directory Layout

```
Teachable_Machine_Replica/
├── image_classifier/
│   ├── color_model.h5           # CNN for image classification
│   └── labels.json              # Label map for image classes
├── pose_classifier/
│   ├── pose_model.keras         # DNN or ML model for pose classification
│   └── pose_labels.npy          # Label map (numpy array)
├── pose_data/                   # Collected pose CSVs
├── data/                        # Collected image samples
├── main.py                      # GUI and app entry point
├── trainer.py                   # Training utilities
├── predictor.py                 # Live prediction logic
```

---

Requirements

* Python 3.10+
* `opencv-python`, `mediapipe`, `tensorflow`, `sklearn`, `joblib`, `catboost`, `tkinter`, `PIL`

