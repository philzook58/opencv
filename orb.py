import cv2
import numpy as np


orb = cv2.ORB()

cap = cv2.VideoCapture(0)

while(1):

    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp = orb.detect(gray,None)
    kp, des = orb.compute(gray, kp)
    img2 = cv2.drawKeypoints(frame,kp,color=(0,255,0), flags=0)
    cv2.imshow('image',cv2.pyrDown(img2))

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
