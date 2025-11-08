import cv2
import os
import mediapipe as mp
import numpy as np

# Environment Setup

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 150)


# Initialize Mediapipe

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define the folder where the color images are stored
folder = 'colors'
mylist = os.listdir(folder)
overlist = []

# Load images from the folder and append them to the list
for i in mylist:
    image = cv2.imread(f'{folder}/{i}')
    print(image.shape)
    overlist.append(image)

# Set the initial header image from the first image in the list
header = overlist[0]

# Create a blank canvas to draw on
canvas = np.zeros((480, 640, 3), np.uint8)

# Basic UI + Hand Tracking

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Add the header (color selection) at the top of the frame
    frame[0:100, 0:640] = header
    
    cv2.imshow('canvas', canvas)
    cv2.imshow("Hand Tracking", frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
