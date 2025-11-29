import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from model import build_model
from preprocess import preprocess

DATA_PATH = "dataset/Train/"
images = []
labels = []

print("Loading dataset...")

for class_id in range(44):
    class_path = os.path.join(DATA_PATH, str(class_id))
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (32, 32))
        img = preprocess(img)

        images.append(img)
        labels.append(class_id)

images = np.array(images)
labels = np.array(labels)

print("Dataset loaded. Total images:", len(images))

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

model = build_model()
print("Training model...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64
)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/traffic_sign_model.h5")

print("Model saved to model/traffic_sign_model.h5")
