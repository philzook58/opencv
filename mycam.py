
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