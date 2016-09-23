import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cv2.namedWindow('image')
def nothing(x):
    pass
cv2.createTrackbar('upper','image',200,255,nothing)
cv2.createTrackbar('lower','image',100,255,nothing)
while(1):

    # Take each frame
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    upper = cv2.getTrackbarPos('upper','image')
    lower = cv2.getTrackbarPos('lower','image')
    edges = cv2.Canny(gray,lower,upper)
    #ret,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
    empty = np.zeros(frame.shape, dtype=np.uint8)
    kernel = np.ones((5,5),np.uint8)
    #edges = cv2.dilate(edges,kernel,iterations = 2) # really chunks it up

    contours,hierarchy= cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(empty, contours, -1, (0,0,255), 3)
    cv2.imshow('res',cv2.pyrDown(empty))
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
