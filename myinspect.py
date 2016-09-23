import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
time.sleep(1)

_, frame = cap.read()

def checkStuff(event,x,y,flags,param):
    print "yo"
    print event
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print frame[x,y,:]
    if event == 0:
        print frame[y,x,:]
cv2.namedWindow('frame')
cv2.imshow('frame',frame)

cv2.setMouseCallback('frame',checkStuff)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
while 1:
    _, frame = cap.read()
    cv2.imshow('frame',frame)
    '''
