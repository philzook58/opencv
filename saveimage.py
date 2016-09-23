import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
time.sleep(1)
print cap.get()
_, frame = cap.read()

cv2.imwrite('myimage.png',frame)
