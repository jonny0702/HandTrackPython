import cv2
import time
import numpy as np
import mediapipe as mp
import math
import HandTrackingModule as htm
##Library to change the volume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#############################
wCam, hCam = 640, 480
#############################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detectorHand = htm.handDetector(minDetectionConfidence=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
   IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

volume = cast(interface, POINTER(IAudioEndpointVolume))   
volumeRange = volume.GetVolumeRange()
volumeControl = 0
volumeBar= 0


#volumeRange = [-65.25, 0.0, 0.03125]
minVolume = volumeRange[0]
maXVolume = volumeRange[1]

def lineLength(x1, x2, y1, y2):
  return math.hypot(x2 - x1, y2 - y1)

while True:
    success, img = cap.read()
    img = detectorHand.findHands(img)
    lmList = detectorHand.findPosition(img, draw=False)

    if len(lmList) != 0 : 
      # print(lmList[4], lmList[8])
      x1, y1 = lmList[4][1], lmList[4][2]
      x2, y2 = lmList[8][1], lmList[8][2]
      ##The center of the Line
      cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
      ##DRAWING THE 4, 8 NODES IN THE HAND
      cv2.circle(img, (x1, y1), 10, (231, 231, 231), cv2.FILLED)   
      cv2.circle(img, (x2, y2), 10, (231, 231, 231), cv2.FILLED)   
      cv2.line(img, (x1,y1), (x2,y2), (231,231,231), 2)
      cv2.circle(img, (cx, cy), 5,(231,231,231), cv2.FILLED)
      ##Getting the distance between the points
      lineDistance = lineLength(x1, x2, y1, y2)

      if lineDistance < 50:
        cv2.circle(img, (cx, cy), 5,(0,255,0), cv2.FILLED)
      #HAND RANGE MIN= 15 MAX= 300
      #VolumeRange -65 - 0
      volumeControl = np.interp(lineDistance,[50,300],[minVolume, maXVolume])
      volumeBar = np.interp(lineDistance,[50,300],[400, 150])
      volume.SetMasterVolumeLevel(volumeControl, None)
    
    cv2.rectangle(img,(50,150), (85,400), (255,255,255), 2)
    cv2.rectangle(img,(50,int(volumeBar)), (85,400), (255,255,255), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime -pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.imshow("Img", img)
    cv2.waitKey(1)


