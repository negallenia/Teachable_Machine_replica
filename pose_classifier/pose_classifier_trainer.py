import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import glob
import os

# Load pose data
files = glob.glob('pose_data/*.csv')
X, y = [], []

for file in files:
    data = pd.read_csv(file, header=None).values.flatten()
    X.append(data)
    label = os.path.basename(file).split('_')[0]
    y.append(label)

X = np.array(X)
y = np.array(y)

# Encode labels as integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build a simple MLP model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Accuracy: {test_acc * 100:.2f}%")

# Optional: save model and label encoder
model.save("pose_classifier/pose_model.keras")
np.save("label_classes.npy", le.classes_)
