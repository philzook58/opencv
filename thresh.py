import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #ret,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
    #ret,thresh = cv2.threshold(gray,50,255,cv2.ADAPTIVE_THRESH_MEAN_C)
    #ret,thresh = cv2.threshold(gray,50,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C )
    ret2,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('res',cv2.pyrDown(thresh))
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
