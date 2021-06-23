import cv2
import time
import os
import numpy as np
import mediapipe as mp
import HandTrackingModule as htm


# Setting the width and height of the camera
wCam, hCam = 1280, 720

# Using Camera "1" for video capture most will need to use camera "0"
cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)

# Object instance of htm
detector = htm.handDetector(max_hands=1, detection_confidence=0.75)

# Getting Images
fingerImgPath = "Fingers"
myList = os.listdir(fingerImgPath)
print(myList)

overlayList = []

# Appending images into a list for easy access
for jpg in myList:
    images = cv2.imread(f'{fingerImgPath}/{jpg}')
    overlayList.append(images)

# Creating a list of fingertip landmark ids
tipIds = [4, 8, 12, 16, 20]

# Initializing previous time and current time to calculate fps later on
pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    # Draw on Hand detection
    img = detector.findHands(img)

    # Creating a list of landmarks with their positions
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        fingers = []

        # Four Finger checking if raised
        for id in range(1, 5):
            if lmList[tipIds[id]][2] > lmList[tipIds[id] - 2][2]:
                fingers.append(0)
            else:
                fingers.append(1)

        # Hand Handedness
        if lmList[tipIds[1]][1] < lmList[tipIds[4]][1]:
            # Right Hand
            Hand = 0

        else:
            # Left Hand
            Hand = 1

        # Checking Thumb
        if lmList[tipIds[0]][1] > lmList[5][1]:
            if Hand == 1:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if Hand == 0:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        totalFingers = fingers.count(1)
        
        # Overlaying our image into the video
        img[0:240, 0:189] = overlayList[totalFingers]

    # Calculating FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Outputting fps at 600x30 with font hershey, size 2, green, 2 thickness
    cv2.putText(img, str(int(fps)), (1200, 30), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)