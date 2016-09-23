import numpy as np


cube =[[i,j,k] for i in range(2) for j in range(2) for k in range(2)]

print cube 




t = np.array([0,0,1])


class Quaternion():
	def __init__(self, axis=np.array([1.,0.,0.]), angle = 0.):
		self.vector = np.sin(angle/2.) * axis / np.linalg.norm(axis)
		self.scalar = np.cos(angle/2.)
	def __init__(self, scalar, vector):
		self.vector = vector
		self.scalar = scalar

	def inv(self):
		self.angle

	def __mul__(self,b):
		newscalar = self.scalar * b.scalar - np.dot(self.vector, b.vector)
		newvector = np.cross(self.vector,b.vector)
		return Quaternion(scalar=newscalar, vector=newvector)


Quaternion()


