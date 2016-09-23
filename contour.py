import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,0)
    kernel = np.ones((3,3),np.uint8)
    #erosion = cv2.erode(gray,kernel,iterations = 1)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(closing,100,200)

    #edges = cv2.cornerHarris(gray, 5)
    #ret,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)

    empty = np.zeros(frame.shape, dtype=np.uint8)


    #kernel = np.ones((5,5),np.uint8)
    #edges = cv2.dilate(edges,kernel,iterations = 2) # really chunks it up

    #contours,hierarchy= cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(empty, contours, -1, (0,0,255), 3)
    #cv2.imshow('res',cv2.pyrDown(empty))
    cv2.imshow('gray',cv2.pyrDown(gray))
    cv2.imshow('res',cv2.pyrDown(opening))
    #cv2.imshow('res',cv2.pyrDown(edges))

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
