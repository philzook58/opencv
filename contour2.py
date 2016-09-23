import cv2
import numpy as np

cap = cv2.VideoCapture(0)


#Noise is fluctuating. Maybe take a couple frames (3 or 4)
#Threshold For being in 2/3 of them.
copies = 4
N=1 #Downsamples
_, frame = cap.read()
_, frame = cap.read()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

for i in range(N):
    gray = cv2.pyrDown(gray)
mybuffer = np.zeros((gray.shape[0],gray.shape[1],copies)


for i in range(copies):
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for i in range(N):
        gray = cv2.pyrDown(gray)
    mybuffer[:,:,i] = gray


currentindex = 0
while(1):

    # Take each frame
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    N=1
    for i in range(N):
        gray = cv2.pyrDown(gray)

    #gray = cv2.pyrDown(gray)
    #gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,0)
    kernel = np.ones((2,2),np.uint8)
    #erosion = cv2.erode(gray,kernel,iterations = 1)
    #blur = cv2.GaussianBlur(gray,(11,11),0)
    thres = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)

    #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    closing = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(closing,100,200)

    #edges = cv2.cornerHarris(gray, 5)
    #ret,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)

    empty = np.zeros(frame.shape, dtype=np.uint8)


    #kernel = np.ones((5,5),np.uint8)
    #edges = cv2.dilate(edges,kernel,iterations = 2) # really chunks it up

    #contours,hierarchy= cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(empty, contours, -1, (0,0,255), 1)
    #cv2.imshow('res',cv2.pyrDown(empty))
    show = closing
    for i in range(N-1):
        show = cv2.pyrUp(show)
    cv2.imshow('gray',gray)
    cv2.imshow('gld',show)
    #cv2.imshow('contours',empty)
    #cv2.imshow('res',cv2.pyrDown(edges))

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
