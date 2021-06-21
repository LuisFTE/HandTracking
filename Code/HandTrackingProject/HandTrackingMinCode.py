import cv2
import mediapipe as mp
import time

# Using Camera "1" for video capture most will need to use camera "0"
cap = cv2.VideoCapture(1)

# Initialize mp to detect hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 1)
mpDraw = mp.solutions.drawing_utils

# Initializing previous time and current time to calculate fps later on
pTime = 0
cTime = 0

# Read camera feed
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Convert BGR image into RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process image using mp.Hands
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    # if the detection is detecting landmarks
    if results.multi_hand_landmarks:
        # We will loop to see how many hands are being detected
        for handLandmarks in results.multi_hand_landmarks:

            # We are looping to identify the id of the landmark and the position in pixels
            for id, landmark in enumerate(handLandmarks.landmark):

                # Height, Width, Color Channel = image shape (height and width in pixels
                h, w, c = img.shape
                print(landmark.z*-1)

                # Calculate the pixel location of landmarks (landmark.x and .y are the x y ratio of the image we are
                # multiplying by the actual size to find the pixel location)
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                # print(id, cx, cy)

                # We are drawing yellow circles on the landmarks 4, 8, 12, 16, and 20 (the tips of the fingers)
                if (id == 4 or id == 8 or id == 12 or id == 16 or id == 20) and (landmark.z*-1) < .1:
                    cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)
                    cv2.putText(img, "Hand is far", (0, 400), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                if (id == 4 or id == 8 or id == 12 or id == 16 or id == 20) and ((landmark.z*-1) >= .1 and (landmark.z*-1) <= .18):
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, "Hand is at a good view!", (0, 425), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

                if (id == 4 or id == 8 or id == 12 or id == 16 or id == 20) and (landmark.z*-1) > .18:
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, "Hand is very close", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)

            # Draw the landmarks on img (We are drawing on the img and not on the rgb because we are outputing the img
            # not the RGBimg)
            # handLandmarks is the points and the mp.Hands.HAND_CONNECTIONS are the connecting lines
            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)

    # Calculating FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # Outputting fps at 600x30 with font hershey, size 2, green, 2 thickness
    cv2.putText(img, str(int(fps)), (600, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Display image
    cv2.imshow("Image", img)
    cv2.waitKey(1)