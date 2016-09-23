import cv2
import random

import numpy as np
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cam = cv2.VideoCapture(0)
while True:
    s, img = cam.read() # captures image



    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )

    img_equ = cv2.equalizeHist(img_gray)

    faces = face_cascade.detectMultiScale(img_equ, 1.3, 5)
    height, width = img_gray.shape


    (x,y,w,h) = faces[0]
    img_face = img_gray[int(max(y-(0.35*h),0)):int(min(y+1.15*h, height)), int(max(x-(0.15*w),0)):int(min(x+1.15*w,width))]


    img_edges = cv2.Canny(img_face, 40, 80)

    contours, hierarchy =  cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    long_contours = []
    for contour in contours:
        if len(contour) > 20:
            long_contours.append(contour)


    #target = open('out.gcode', 'w')

    blank = np.zeros(img_edges.shape)
    blank[:] = 255
    cv2.drawContours(blank, long_contours, -1, (0,255,0), 1)
    #cv2.drawContours(img_face, contours, -1, (0,255,0), 1)
    cv2.imshow("Test Picture", blank)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
pen_down = False


for contour in contours:
    for point in contour:
        target.write("G1 X" + str(point[0][0] * 800 / width) + " Y" + str(point[0][1] * 600 / height) + "\n")
        if not pen_down:
            target.write("M03\n")
            pen_down = True
    target.write("M05\n")
    pen_down = False

'''
'''
for contour in contours:
    for point in contour:
        print point[0]
    print
'''
