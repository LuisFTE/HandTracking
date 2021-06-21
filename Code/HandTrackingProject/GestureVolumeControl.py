import math
import cv2
import time
import numpy as np
import mediapipe as mp
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Setting the width and height of the camera
wCam, hCam = 1280, 720

# Using Camera "1" for video capture most will need to use camera "0"
cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(max_hands=2, detection_confidence=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volumeRange = volume.GetVolumeRange()


minVol = volumeRange[0]
maxVol = volumeRange[1]
bar_vol = 400
percent_vol = 0

# Initializing previous time and current time to calculate fps later on
pTime = 0
cTime = 0

# Read camera feed
while True:
    success, img = cap.read()

    # Detect Hands
    img = detector.findHands(img)

    # Find pixel coordinates of index and thumb
    lmklist = detector.findPosition(img, draw=False)

    # Only when the hand is detected
    if len(lmklist) != 0:

        # Label the x, y coordinates of the index and thumb
        x1, y1 = lmklist[4][1], lmklist[4][2]
        x2, y2 = lmklist[8][1], lmklist[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw a circle on the targeted fingers
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)

        # Drawing line between target fingers
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        # Hand Range is 50 -> 300
        # Volume Range -65 -> 0

        vol = np.interp(length, [50, 350], [minVol, maxVol])
        bar_vol = np.interp(length, [50, 350], [400, 150])
        percent_vol = np.interp(length, [50, 350], [0, 100])

        volume.SetMasterVolumeLevel(vol, None)

        print(vol, int(length))

        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(bar_vol)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(percent_vol)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 0), 2)





    # Calculating FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Outputting fps at 600x30 with font hershey, size 2, green, 2 thickness
    cv2.putText(img, str(int(fps)), (1200, 30), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 0), 2)

    # Display image
    cv2.imshow("Image", img)
    cv2.waitKey(1)

