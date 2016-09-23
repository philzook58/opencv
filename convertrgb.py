#RGB
import cv2
import numpy as np
green = np.uint8([[[0,255,0 ]]])
hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
print hsv_green
lower_hsv = [hsv_green[0,0,0]-10,50,50]
upper_hsv = [hsv_green[0,0,0]+10,255,255]
print lower_hsv
print upper_hsv
