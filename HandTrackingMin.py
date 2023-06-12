import cv2
import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time 

cap = cv2.VideoCapture(0)


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0

cTime = 0

while True: 
    success, img = cap.read()
    ##Converting the image to RGB image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB) 

    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #We Traverse the Array of data and we print each landMark
            # The id is the nodes of the hand and the lm is the coordinates in the 3d Space
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                #Finding the location of the id and landmarks using with and height
                height, wth, chanels = img.shape
                ##The position x and y
                cx, cy = int(lm.x * wth), int(lm.y * height)
                print(id, cx, cy)
                if(id == 4):
                    cv2.circle(img, (cx, cy), 10, (255, 100, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    ##FPS Time Screen
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)


