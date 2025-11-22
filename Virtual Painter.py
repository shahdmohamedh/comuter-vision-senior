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

# Drawing State Variables
draw_color = (0, 0, 0)
brush_thickness = 10
xp, yp = 0, 0 # Previous x, y coordinates for smooth line drawing

# Define the folder where the color images are stored
folder = 'colors'
mylist = os.listdir(folder)
overlist = []

# Utility Function for Coordinate Extraction 
def find_position(img, hand_landmarks, lm_num):
    h, w, c = img.shape
    x, y = 0, 0
    
    if hand_landmarks:
        lm = hand_landmarks.landmark[lm_num]
        x, y = int(lm.x * w), int(lm.y * h)
    
    return x, y

# Load images from the folder and append them to the list
for i in mylist:
    image = cv2.imread(f'{folder}/{i}')
    overlist.append(image)

header = overlist[0]

canvas = np.zeros((480, 640, 3), np.uint8)

SECTION_WIDTH = 80 # 640 / 8 = 80

# List of tuples: (start_x, end_x, header_index, BGR_color, thickness)
COLOR_OPTIONS = [
    # Section 1: Palette/Brush
    (7 * SECTION_WIDTH, 8 * SECTION_WIDTH, 0, None, 10), 
    # Section 2: Eraser
    (0 * SECTION_WIDTH, 1 * SECTION_WIDTH, 7, (0, 0, 0), 50), 
    # Section 3: Purple Splash
    (1 * SECTION_WIDTH, 2 * SECTION_WIDTH, 6, (255, 82, 140), 10), 
    # Section 4: White Splash 
    (2 * SECTION_WIDTH, 3 * SECTION_WIDTH, 5, (255, 255, 255) , 10), 
    # Section 5: Green Splash
    (3 * SECTION_WIDTH, 4 * SECTION_WIDTH, 4, (86, 166, 0), 10), 
    # Section 6: Yellow Splash
    (4 * SECTION_WIDTH, 5 * SECTION_WIDTH, 3, (28, 238, 250), 10), 
    # Section 7: Blue Splash
    (5 * SECTION_WIDTH, 6 * SECTION_WIDTH, 2, (222, 161, 0), 10), 
    # Section 8: Red Splash
    (6 * SECTION_WIDTH, 7 * SECTION_WIDTH, 1, (0, 0, 255), 10), 
]

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        xp, yp = 0, 0 
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            x1, y1 = find_position(frame, hand_landmarks, 8)
            x2, y2 = find_position(frame, hand_landmarks, 12)
            _, y6 = find_position(frame, hand_landmarks, 6)
            
            if y1 < y6 and y2 > y6: 
                mode = 'Selection'
                cv2.circle(frame, (x1, y1), 10, (255, 255, 255), cv2.FILLED) 
                xp, yp = 0, 0
                
                if y1 < 100: 
                    for start_x, end_x, h_idx, color, thick in COLOR_OPTIONS:
                        if start_x < x1 < end_x and color is not None:
                            header = overlist[h_idx]
                            draw_color = color
                            brush_thickness = thick
                            break

            elif y1 < y6: 
                mode = 'Drawing'
                cv2.circle(frame, (x1, y1), brush_thickness, draw_color, cv2.FILLED)

                if xp == 0 and yp == 0:
                    xp, yp = x1, y1 
                
                cv2.line(canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
                
                xp, yp = x1, y1
            
            else:
                mode = 'Rest'
                xp, yp = 0, 0 

    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _,  mask = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY) 
    mask_inv = cv2.bitwise_not(mask) 
    mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
    
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask  = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    frame_fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    
    canvas_bg = cv2.bitwise_and(canvas, canvas, mask=mask)
    final_image = cv2.add(frame_fg, canvas_bg) 
    
    final_image[0:100, 0:640] = header
    
    cv2.imshow("Virtual Painter", final_image) 
    cv2.imshow('Canvas', canvas)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros((480, 640, 3), np.uint8)
    elif key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()