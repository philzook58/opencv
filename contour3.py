import cv2
import numpy as np

cap = cv2.VideoCapture(0)


#Noise is fluctuating. Maybe take a couple frames (3 or 4)
#Threshold For being in 2/3 of them.
copies = 2
N=1 #Downsamples




def magic(frame):
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
        #thres = cv2.dilate(thres,kernel,iterations = 1)

        #closing = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)
        #opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)


        #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        #edges = cv2.Canny(closing,100,200)
        return thres

_, frame = cap.read()
gray = magic(frame)
mybuffer = np.zeros((gray.shape[0],gray.shape[1],copies))


for i in range(copies):
    _, frame = cap.read()

    thres = magic(frame)
    mybuffer[:,:,i] = gray


currentindex = 0

while(1):

    # Take each frame
    _, frame = cap.read()

    thres = magic(frame)
    mybuffer[:,:,currentindex] = thres
    #kernel = np.ones((2,2),np.uint8)
    #thres = cv2.dilate(thres,kernel,iterations = 1)
    #edges = cv2.cornerHarris(gray, 5)
    #ret,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)

    empty = np.ones(thres.shape, dtype=np.uint8)
    empty = mybuffer[:,:,0]
    for i in range(copies):
        empty = cv2.bitwise_or(empty,mybuffer[:,:,i])
    empty = empty.astype(np.uint8)
    #print empty
    kernel = np.ones((2,2),np.uint8)
    #empty = cv2.morphologyEx(empty, cv2.MORPH_OPEN, kernel)
    '''
    empty = cv2.GaussianBlur(empty,(11,11),0)
    ret,empty = cv2.threshold(empty,200,255,cv2.THRESH_BINARY)
    '''
    #edges = cv2.dilate(edges,kernel,iterations = 2) # really chunks it up
    #closing = cv2.morphologyEx(empty, cv2.MORPH_CLOSE, kernel)
    contours,hierarchy= cv2.findContours(empty,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cnt = contours[0]
    #print contours
    i=0
    area = np.zeros(len(contours))
    for cnt in contours:

        area[i] = cv2.contourArea(cnt)
        i=i+1
    #print area
    indexorder= np.argsort(area)[::-1]
    #print indexorder


    newempty = np.ones((thres.shape[0],thres.shape[1],3), dtype=np.uint8)
    fraction = 16
    print len(indexorder)/fraction
    for i in range(len(indexorder)/fraction):

        #cv2.drawContours(newempty, contours, indexorder[i], (0,0,255), 1)
        #Simplified triangles

        cnt = contours[indexorder[i+3]]
        epsilon = 0.02*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        cv2.drawContours(newempty, [approx], 0, (0,255,0), 1)

        #convex
        """
        cnt = contours[indexorder[i+10]]
        hull = cv2.convexHull(cnt)
        cv2.drawContours(newempty, [hull], 0, (0,255,0), 1)
        """

    #cv2.imshow('res',cv2.pyrDown(empty))
    show = newempty
    for i in range(N-1):
        show = cv2.pyrUp(show)

    cv2.imshow('gray',thres)
    cv2.imshow('gld',show)
    #cv2.imshow('contours',empty)
    #cv2.imshow('res',cv2.pyrDown(edges))

    currentindex = currentindex + 1
    currentindex = currentindex % copies
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
