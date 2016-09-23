import cv2
import numpy as np



class MyCam():
	def __init__(self, frameSize=(480,640), focus =600, avgPointPos=np.array([0,0,0]), sigma = 3, pointNum=1000):
		self.pointCloud = sigma * np.random.randn(pointNum, 3)
		self.pointCloud = map(lambda pnt: pnt + avgPointPos, self.pointCloud)
		self.t = np.zeros(3)
		self.R = np.identity(3)
		self.frameSize = frameSize
		self.focus = focus
	def read(self):
		pnts = self.projectPoints()
		mask = self.mask()
		pnts = pnts[mask==True]
		frame = np.zeros(self.frameSize + (3,))

		for pnt in pnts.astype(int):
			if pnt[0] > 0 and pnt[1] > 1 and pnt[0] < self.frameSize[0] and pnt[1] < self.frameSize[1]:
				cv2.circle(frame,tuple(pnt),5,[0,0,255],-1)
		return frame
	def addVecToPoint(self,points,vec):
		return map(lambda pnt: pnt + vec, points)
	def transformPoints(self):
		rotated = np.dot(self.pointCloud, self.R.T)
		translated = self.addVecToPoint(rotated, self.t)
		return translated
	def mask(self):
		transformed = self.transformPoints()
		return np.array(map(lambda pnt: pnt[2] > 0, transformed))
	def projectPoints(self):
		transformed = self.transformPoints()
		return np.array(map(lambda pnt: self.focus * pnt[:2]/pnt[2] + np.array(self.frameSize)/2, transformed))


def drawRotate(w,h,rotation, img):

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

def calculateCamera(w,h, p_old, p_new,old_mask,new_mask):

	'''
	def cleanPoints(pts):
		mean = np.sum(pts, axis = 0)
		pts = map(lambda pt: pt - mean, pts)
		scale = np.sum(np.linalg.norm(pts, axis=2)) / len(pts)
		pts = pts * np.sqrt(2) /scale
		return mean, scale, pts
		'''
	#mean, scale, p_old = cleanPoints(p_old)
	def reformat(pnts):
		#print pnts
		pnts = np.array(pnts)
		pnts = pnts[np.logical_and(new_mask, old_mask)]
		return np.array(map(lambda pnt: [pnt], pnts))# This wrapping should be turned into a reshape call
	p_old = reformat(p_old)
	p_new = reformat(p_new)

	def normPts(pts):
		pts[:,:,0] = pts[:,:,0]/w - 0.5 
		pts[:,:,1] = pts[:,:,1]/h - 0.5
		return pts
	#print p_old.shape
	p_old = normPts(p_old)
	p_new = normPts(p_new)

	#F, mask = cv2.findFundamentalMat(p_old,p_new,cv2.FM_RANSAC) #F p_old = line in p_new
	F, mask = cv2.findEssentialMat(p_old,p_new) #F p_old = line in p_new
	p_old = p_old[mask==1]
	p_new = p_new[mask==1]
	U, s, V = np.linalg.svd(F) #, full_matrices=True) #Note that V here is often called V.H eslewhere in literature
	#F = np.dot(np.dot(U, np.diag([1,1,0])),V)
	retval, R, t, mask = cv2.recoverPose(F, p_old, p_new)

	#print t

	if retval < 10:
		R=np.identity(3)
	#print mask
	return R, t[0]


cam = MyCam()
'''
frame = cam.read()
cv2.imshow('frame',frame)
k = cv2.waitKey(0)
'''

angle = .1

rotateZ = np.array([[np.cos(angle), np.sin(angle), 0],
					[-np.sin(angle), np.cos(angle), 0],
					[0,0,1]])

rotateX = np.array([[1,0,0],
					[0,np.cos(angle), np.sin(angle)],
					[0,-np.sin(angle), np.cos(angle)]])

old_pnts= cam.projectPoints()
old_mask = cam.mask()
R = np.identity(3)
while(1):
	frame = cam.read()
	new_pnts = cam.projectPoints()
	new_mask = cam.mask()

	rotate, t = calculateCamera(640,480,old_pnts,new_pnts,old_mask,new_mask)
	R = np.dot(rotate,R)
	frame = drawRotate(640,480,R, frame)
	cam.R = np.dot(rotateX, cam.R)

	cv2.imshow('frame',frame)
	old_pnts = new_pnts
	old_mask = new_mask
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

rotateX = np.array([[1,0,0],
					[0,np.cos(angle), np.sin(angle)],
					[0,-np.sin(angle), np.cos(angle)]])
	
cv2.destroyAllWindows()
