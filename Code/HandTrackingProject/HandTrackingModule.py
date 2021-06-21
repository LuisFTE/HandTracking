import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.maxHands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detection_confidence, self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        # Convert BGR image into RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process image using mp.Hands
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        # if the detection is detecting landmarks
        if self.results.multi_hand_landmarks:
            # We will loop to see how many hands are being detected
            for handLandmarks in self.results.multi_hand_landmarks:
                # Draw the landmarks on img (We are drawing on the img and not on the rgb because we are
                # outputting the img not the RGBimg)
                # handLandmarks is the points and the mp.Hands.HAND_CONNECTIONS are the connecting lines
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        landmarklist = []

        if self.results.multi_hand_landmarks:

            myHand = self.results.multi_hand_landmarks[handNo]

            for id, landmark in enumerate(myHand.landmark):
                # Height, Width, Color Channel = image shape (height and width in pixels
                h, w, c = img.shape

                # Calculate the pixel location of landmarks (landmark.x and .y are the x y ratio of the image we are
                # multiplying by the actual size to find the pixel location)
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                # print(id, cx, cy)
                landmarklist.append([id, cx, cy])

                if draw:
                    if (id == 4 or id == 8 or id == 12 or id == 16 or id == 20) and (landmark.z*-1) < .1:
                        cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)
                        cv2.putText(img, "Hand is far", (0, 630), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                    if (id == 4 or id == 8 or id == 12 or id == 16 or id == 20) and (.1 <= (landmark.z * -1) <= .18):
                        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, "Hand is at a good view!", (0, 645), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

                    if (id == 4 or id == 8 or id == 12 or id == 16 or id == 20) and (landmark.z*-1) > .18:
                        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
                        cv2.putText(img, "Hand is very close", (0, 700), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)

        return landmarklist


def main():

    # Using Camera "1" for video capture most will need to use camera "0"
    cap = cv2.VideoCapture(1)

    detector = handDetector(max_hands=1)

    # Initializing previous time and current time to calculate fps later on
    pTime = 0
    cTime = 0

    # Read camera feed
    while True:
        success, img = cap.read()

        img_flip = cv2.flip(img, 1)
        img = detector.findHands(img_flip)
        lmList = detector.findPosition(img_flip)

        if len(lmList) != 0:
            print(lmList[8])

        # Calculating FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Outputting fps at 600x30 with font hershey, size 2, green, 2 thickness
        cv2.putText(img, str(int(fps)), (600, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Display image
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()