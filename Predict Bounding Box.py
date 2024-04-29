from tensorflow.keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model("/home/ayman/model.h5")

# Load and preprocess the single image
image_path = "/mnt/c/Users/Ayman/Downloads/8.jpg"

# Specify the target size based on the model's input shape
target_size = model.input_shape[1:3]

# Load and resize the image to the target size
img = image.load_img(image_path, target_size=target_size)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make predictions on the single image
prediction = model.predict(img_array)[0]

# Denormalize bounding box coordinates
prediction[:2] *= target_size[0]  # x, y coordinates
prediction[2:] *= target_size[0]  # width, height

# Draw bounding box on the image
x, y, w, h = prediction
x, y, w, h = int(x), int(y), int(w), int(h)

img = cv2.imread(image_path)
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with bounding box
cv2.imshow("Image with Bounding Box", img)
cv2.waitKey(0)
cv2.destroyAllWindows()