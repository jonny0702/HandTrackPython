import cv2
import mediapipe as mp 
import time 

class handDetector():
    def __init__(self, mode=False, maxHands=2, minDetectionConfidence=0.5, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectConf = minDetectionConfidence
        self.trackConf = trackConf
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, int(self.detectConf), self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    #Detection
    def findHands(self, img, draw=True):
        ##Converting the image to RGB image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) 
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw : 
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    #find the position
    def findPosition(self, img, handNumber=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            #Selecting a particular Hand
            myHand = self.results.multi_hand_landmarks[handNumber]
            # The id is the nodes of the hand and the lm is the coordinates in the 3d Space
            for id, lm in enumerate(myHand.landmark):
                #Finding the location of the id and landmarks using with and height
                height, wth, chanels = img.shape
                ##The position x and y
                cx, cy = int(lm.x * wth), int(lm.y * height)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 100, 255), cv2.FILLED)   
        return lmList
    

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True: 
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if (len(lmList)!= 0): print(lmList[4])
        ##FPS Time Screen
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, "fps: " + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
