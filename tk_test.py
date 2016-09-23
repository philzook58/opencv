import cv2
import numpy as np




from Tkinter import *

master = Tk()
minval = Scale(master, from_=0, to=255,  orient=HORIZONTAL, label="min")
minval.pack()
maxval = Scale(master, from_=0, to=255, orient=HORIZONTAL, label="mac")
maxval.pack()


cap = cv2.VideoCapture(0)

while(1):
    #tkinter maintenance routines
    master.update_idletasks()
    master.update()
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray,minval.get(),maxval.get())
    cv2.imshow('image',cv2.pyrDown(edges))

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
