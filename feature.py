import cv2
import numpy as np


cap = cv2.VideoCapture(0)


while(1):

    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray,1000,0.01,10)
    corners = np.int0(corners)


    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

     # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    for i in corners:
        x,y = i.ravel()
        cv2.circle(frame,(x,y),8,[0,0,255],-1) #image center radius color thickness
    cv2.imshow('image',cv2.pyrDown(frame))

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
