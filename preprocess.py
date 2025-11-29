import cv2
import numpy as np

def preprocess(img):
    # Convert BGR to Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Histogram equalization (improves contrast)
    img = cv2.equalizeHist(img)

    # Normalize 0-1
    img = img / 255.0

    # Resize into 32x32x1 format
    img = np.reshape(img, (32, 32, 1))

    return img
