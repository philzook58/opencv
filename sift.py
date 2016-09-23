import cv2
import numpy as np


sift = cv2.xfeatures2d.SIFT_create()

cap = cv2.VideoCapture(0)


while(1):

    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray,None)
    img=cv2.drawKeypoints(frame,kp)
    cv2.imshow('image',cv2.pyrDown(frame))

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
