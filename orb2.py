import cv2
import numpy as np


orb = cv2.ORB()

minpoint = []

def findNearestKeyPoint(x,y,kp):
    global minpoint
    mindist = 100000000000
    minpoint = kp[0]
    for point in kp:
        dist = (point.pt[0]-x)**2+(point.pt[1]-y)**2
        if dist < mindist:
            mindist = dist
            minpoint = point


def service_mouse(event,x,y,flags,param):
    global kp
    if event == cv2.EVENT_LBUTTONDOWN:
        findNearestKeyPoint(x,y,kp)
        #mycolor = np.uint8([[cv2.pyrDown(frame)[y,x]]])
        #print mycolor
        #color = cv2.cvtColor(mycolor, cv2.COLOR_BGR2HSV)




cap = cv2.VideoCapture(0)

cv2.namedWindow('image')
cv2.setMouseCallback('image',service_mouse)

while(1):

    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp = orb.detect(gray,None)
    kp, des = orb.compute(gray, kp)
    img2 = cv2.drawKeypoints(frame,kp,color=(0,255,0), flags=0)
    if minpoint:
        cv2.circle(img2, (int(minpoint.pt[0]),int(minpoint.pt[1])) , 4, [0,0,255],-1 )
    cv2.imshow('image',cv2.pyrDown(img2))

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
