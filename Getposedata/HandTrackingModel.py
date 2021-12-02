import cv2
import time
import mediapipe as mp


class HandDetector():
    def __init__(self, model=False, maxnumhand=2, detectionCon=0.5, trackCon=0.5):
        self.model = model
        self.maxnumhand = maxnumhand
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.model, self.maxnumhand, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  # detect the Hands

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:  # total 21 marks
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, HandNum=0, draw=False):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myhandLms = self.results.multi_hand_landmarks[HandNum]
            for id, lm in enumerate(myhandLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, ':', cx, cy)
                lmlist.append([id, cx, cy])
                if draw:
                    if id == 17:
                        cv2.circle(img, (cx, cy), 15, (0, 255, 255), cv2.FILLED)

        return lmlist


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img2 = detector.findHands(img)
        lmlist = detector.findPosition(img2, 0, True)

        if len(lmlist) != 0:
            print(lmlist[1])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img2, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img2)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()