import numpy as np
import cv2

cap = cv2.VideoCapture(0)

_, frame = cap.read()
_, frame = cap.read()

lk_params = dict( winSize  = (15,15),
			   maxLevel = 2,
				criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def reset():
	_, frame = cap.read()

	old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	p0 = cv2.goodFeaturesToTrack(old_gray,300,0.01,10)
	return old_gray, p0

def update(old_gray, p0):
	_, frame = cap.read()
	new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)
	good_new = p1[st==1]
	good_old = p0[st==1]
	old_gray = new_gray.copy()
	p0 = good_new.reshape(-1,1,2)
	return frame, new_gray, p0

def draw(frame, pts):
	for pt1 in pts:
		cv2.circle(frame,tuple(pt1),5,[0,0,255],-1)
	cv2.imshow('frame',cv2.pyrDown(frame))

old_gray, p_old = reset()



while True:

	while len(p_old) < 50:
		old_gray, p_old = reset()
	frame, old_gray, p_new = update(old_gray, p_old)
	draw(frame, p_new.reshape(-1,2))

	p_old = p_new

	
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cv2.destroyAllWindows()
cap.release()