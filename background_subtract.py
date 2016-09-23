import numpy as np
import cv2

cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(history=30)
#fgbg = cv2.BackgroundSubtractorMOG()
#fgbg = cv2.createBackgroundSubtractorMOG()
#fgbg = cv2.createBackgroundSubtractorKNN()

'''
    createBackgroundSubtractorKNN(...)
        createBackgroundSubtractorKNN([, history[, dist2Threshold[, detectShadows]]]) -> retval
    
    createBackgroundSubtractorMOG2(...)
        createBackgroundSubtractorMOG2([, history[, varThreshold[, detectShadows]]]) -> retval
'''
for i in range(20):
	ret, frame = cap.read()
	frame = cv2.pyrDown(frame)
	fgmask = fgbg.apply(frame)
	ret,thresh1 = cv2.threshold(fgmask,127,255,cv2.THRESH_BINARY)



y , x =np.mgrid[0:thresh1.shape[0] , 0:thresh1.shape[1]]

xavg=0 
yavg=0

while(1):
	ret, frame = cap.read()
	frame = cv2.pyrDown(frame)
	fgmask = fgbg.apply(frame)
	ret,thresh1 = cv2.threshold(fgmask,127,255,cv2.THRESH_BINARY)

	cv2.imshow('frame',thresh1)
	total = np.sum(thresh1)
	if total > 100:
		yavg = np.sum(thresh1 * y)/total
		xavg = np.sum(thresh1 * x)/total

	thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
	cv2.circle(thresh1,( int(xavg) ,int(yavg) ),5,[0,0,255],-1)
	cv2.imshow('frame',thresh1)


	k = cv2.waitKey(30) & 0xff
	if k == 27:
  		break

cap.release()
cv2.destroyAllWindows()
