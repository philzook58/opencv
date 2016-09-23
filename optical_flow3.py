import numpy as np
import cv2

cap = cv2.VideoCapture(0)

_, frame = cap.read()
_, frame = cap.read()
h = frame.shape[0]
w = frame.shape[1]

lk_params = dict( winSize  = (15,15),
			   maxLevel = 2,
				criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def reset():
	_, frame = cap.read()

	old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	p0 = cv2.goodFeaturesToTrack(old_gray,1000,0.01,10)
	return old_gray, p0

def update(old_gray, p0):
	_, frame = cap.read()
	new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)
	good_new = p1[st==1]
	good_old = p0[st==1]
	old_gray = new_gray.copy()
	p_new = good_new.reshape(-1,1,2)
	p_old = good_old.reshape(-1,1,2)
	return frame, new_gray, p_new, p_old

def draw(frame, pts):
	for pt1 in pts:
		cv2.circle(frame,tuple(pt1),5,[0,0,255],-1)
	return frame


def triangulate(R, t, p1,p2):
	x = np.zeros(3)
	y = np.array([ p1[0] ,p1[1]  , 1 ])
	v = R[1,:] + p2[0] * R[2,:]
	x[2] = np.dot(  v  , t ) / np.dot(v , y)
	x[:2] = x[2] * p1
	print x
	return x


def calculateCamera(p_old, p_new):

	'''
	def cleanPoints(pts):
		mean = np.sum(pts, axis = 0)
		pts = map(lambda pt: pt - mean, pts)
		scale = np.sum(np.linalg.norm(pts, axis=2)) / len(pts)
		pts = pts * np.sqrt(2) /scale
		return mean, scale, pts
		'''
	#mean, scale, p_old = cleanPoints(p_old)
	p_old = p_old.copy()
	p_new = p_new.copy()

	def normPts(pts):
		pts[:,:,0] = pts[:,:,0]/w - 0.5 
		pts[:,:,1] = pts[:,:,1]/h - 0.5
		return pts
	#print p_old.shape
	p_old = normPts(p_old)
	p_new = normPts(p_new)

	F, mask = cv2.findFundamentalMat(p_old,p_new,cv2.FM_RANSAC) #F p_old = line in p_new
	p_old = p_old[mask==1]
	p_new = p_new[mask==1]
	U, s, V = np.linalg.svd(F) #, full_matrices=True) #Note that V here is often called V.H eslewhere in literature
	F = np.dot(np.dot(U, np.diag([1,1,0])),V)
	retval, R, t, mask = cv2.recoverPose(F, p_old, p_new)

	print t

	if retval < 10:
		R=np.identity(3)
	#print mask
	return R, t[0]

def drawRotate(rotation, img):

	corner = (w/2,h/2)
	#rotation = np.rint(rotation * w/5).astype(int)
	vecs = rotation[:,:2] * w/5
	for i in range(3):
		vecs[i,:] = vecs[i,:] + np.array([w/2, h/2])
	vecs = np.rint(vecs).astype(int)
	cv2.line(img, corner, tuple(vecs[0,:]), (255,0,0), 5)
	cv2.line(img, corner, tuple(vecs[1,:]), (0,255,0), 5)
	cv2.line(img, corner, tuple(vecs[2,:]), (0,0,255), 5)
	return img

def drawDirection(t, img):

	corner = (w/2,h/2)
	#rotation = np.rint(rotation * w/5).astype(int)
	vec = t[:2]* w/5 + np.array([w/2, h/2])
	vec = np.rint(vec).astype(int)
	cv2.line(img, corner, tuple(vec), (155,155,0), 5)

	return img


rotation = np.identity(3)
old_gray, p_old = reset()
camera_old = p_old
i =0



while True:

	while len(p_old) < 50:
		old_gray, p_old = reset()
	frame, old_gray, p_new, p_old = update(old_gray, p_old)
	
	#i += 1
	#if i%5 == 0:
	R,t = calculateCamera(p_old, p_new)
	rotation = np.dot(R,rotation)
	#camera_old = p_new
	#print rotation
	frame = drawRotate(rotation, frame)
	frame = drawDirection(t,frame)
	frame = draw(frame, p_new.reshape(-1,2))

	#cv2.line(img, corner, tuple(t[2,:]), (0,0,255), 5)
	cv2.imshow('frame',cv2.pyrDown(frame))

	p_old = p_new

	
	k = cv2.waitKey(100) & 0xff
	if k == 27:
		break

cv2.destroyAllWindows()
cap.release()