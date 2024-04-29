import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from mtcnn.mtcnn import MTCNN
import sys
import cv2

# Function to detect landmarks using MTCNN
def detect_landmarks(image_path):
    pixels = cv2.imread(image_path)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    landmarks = []
    if results:
        landmarks = results[0]['keypoints']
    return landmarks

# Check if GPU is available
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
    sys.exit()

# Load partition information from the text file with space as delimiter
partition_file = "/mnt/c/Users/Ayman/Downloads/New folder/CelebA/Eval/list_eval_partition.txt"
partition_df = pd.read_csv(partition_file, delim_whitespace=True, header=None, names=["image_id", "partition"])

# Specify the path to the CelebA dataset on your computer
celeba_path = "/home/ayman/img_align_celeba"

# Load images and MTCNN landmarks from the CelebA dataset
X, y = [], []
for index, row in partition_df.iterrows():
    filename = row["image_id"]
    img_path = os.path.join(celeba_path, filename)

    # Detect landmarks using MTCNN
    landmarks = detect_landmarks(img_path)

    # Skip images where MTCNN couldn't detect landmarks
    if not landmarks:
        continue

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    X.append(img_array)
    y.append(np.array(landmarks).flatten())

X = np.array(X) / 255.0
y = np.array(y)

# Normalize facial landmarks coordinates to be in the range [0, 1]
y /= 224.0

# Convert target values to float32
y = y.astype(np.float32)

# Split the dataset into training, validation, and testing sets
train_df = partition_df[partition_df["partition"] == 0]
val_df = partition_df[partition_df["partition"] == 1]
test_df = partition_df[partition_df["partition"] == 2]

X_train, y_train = X[train_df.index], y[train_df.index]
X_val, y_val = X[val_df.index], y[val_df.index]
X_test, y_test = X[test_df.index], y[test_df.index]

# Implement a learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9
)

# Build a model with ResNet50 as the feature extractor
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='linear')  # Assuming 5 facial landmarks (x, y for each)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='mean_squared_error',  # Regression loss for facial landmarks coordinates
              metrics=['accuracy'])

# Data Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=2,           # Stop after no improvement in 3 epochs
    restore_best_weights=True  # Restore the weights from the epoch with the best value of the monitored quantity
)

# Train the model with data augmentation for 1 epoch
model.fit(datagen.flow(X_train, y_train, batch_size=64),
          steps_per_epoch=len(X_train) / 64,
          epochs=10,
          validation_data=(X_val, y_val),
          callbacks=[early_stopping],
          verbose=1)

# Save the trained model
model.save("/home/ayman/model_resnet_mtcnn_landmarks.h5")