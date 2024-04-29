import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt

# Load the trained model
loaded_model = load_model("/home/ayman/model_landmarks.h5")

# Load and preprocess the input image
input_image_path = "/mnt/c/Users/Ayman/Downloads/8.jpg"
input_img = image.load_img(input_image_path, target_size=(50, 37))
input_img_array = image.img_to_array(input_img) / 255.0
input_img_array = np.expand_dims(input_img_array, axis=0)

# Predict facial landmarks
predicted_landmarks = loaded_model.predict(input_img_array)

# Post-process the predicted landmarks
predicted_landmarks[:, :10:2] *= 50.0  # x coordinates
predicted_landmarks[:, 1:10:2] *= 37.0  # y coordinates

# Load the original image
original_img = cv2.imread(input_image_path)

# Draw the predicted landmarks on the image
for i in range(5):
    x = int(predicted_landmarks[0, i * 2])
    y = int(predicted_landmarks[0, i * 2 + 1])
    cv2.circle(original_img, (x, y), 2, (0, 255, 0), -1)

# Display the image with landmarks
cv2.imshow("Image with Bounding Box", original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()