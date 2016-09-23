import cv2
import numpy as np


'''
def slider():
    from Tkinter import *

    master = Tk()
    minval = Scale(master, from_=0, to=255)
    minval.pack()
    maxval = Scale(master, from_=0, to=255, orient=HORIZONTAL)
    maxval.pack()
    mainloop()
import threading
t = threading.Thread(target=slider)
t.start()
'''

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('image')
cv2.createTrackbar('Min Edge','image',100,255,nothing)
cv2.createTrackbar('Max Edge','image',200,255,nothing)

while(1):

    # Take each frame
    minval = cv2.getTrackbarPos('Min Edge','image')
    #

    maxval = cv2.getTrackbarPos('Max Edge','image')

    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # adaptive threshold might be good here

    edges = cv2.Canny(gray,minval,maxval)
    cv2.imshow('image',cv2.pyrDown(edges))

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
