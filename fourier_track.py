import cv2
import numpy as np


framenum = 30
cap = cv2.VideoCapture(0)
_, frame = cap.read()


frames = np.zeros( (framenum,) + frame.shape)
for i in range(30):
    _, frame = cap.read()
    frames[i,:,:,:] = frame


print("let's go")
current = 0
while(1):

    _, frame = cap.read()
    frames[current,:,:,:] = frame
    current += 1
    current = current % framenum #Stops index from overflowing over size of array

    fft = np.fft.fft(frames, axis=0)
    fft[0,:,:,:] = 0. #Block off 0 freqeuncy (stationary stuff)

    spectrum = np.sum(np.abs(fft), axis = 0)
    #Naw. I should convert to black and white (or color filter?)
    #Then convert freqeuncy to color
    #Maybe make it so I can click on a point and see the spectrum plot there
    cv2.imshow('res',cv2.pyrDown(spectrum))
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
