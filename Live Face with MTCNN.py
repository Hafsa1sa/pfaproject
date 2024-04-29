import torch
from torchvision import transforms 
from PIL import Image 
import numpy as np 
import cv2 
from mtcnn import MTCNN

cap = cv2.VideoCapture(0)
detector = MTCNN()

# Set a higher frame rate
cap.set(cv2.CAP_PROP_FPS, 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to a smaller resolution
    frame = cv2.resize(frame, (640, 480))

    output = detector.detect_faces(frame)

    for single_output in output:
        x, y, w, h = single_output['box']
        landmarks = single_output['keypoints']

        # Draw bounding box
        cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)

        # Draw facial landmarks
        for point in landmarks.values():
            cv2.circle(frame, center=(int(point[0]), int(point[1])), radius=2, color=(0, 255, 0), thickness=-1)

    cv2.imshow('win', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
