import numpy as np
import scipy as sp
from scipy import optimize
from ad import gh
from mycam import MyCam

def project(pts3):

	return np.array([pts3[:,0] / pts3[:,2], pts3[:,1] / pts3[:,2]]).T


def transform(R,t,pts3):
	return np.dot(pts3, R.T) + t


def pack(R,t,pts3):
	x = np.zeros(R.size + t.size + pts3.size)
	x[0:9] = R.flatten()
	x[9:12] = t
	x[12:] = pts3.flatten()
	return x
def unpack(x):
	return x[0:9].reshape((3,3)), x[9:12], x[12:].reshape((-1,3)) #R,t,pts3

def error_func(ptlist):
	def bundle_error(x):
		R,t,pts3 = unpack(x)
		residual1 = np.sum((project(pts3) - ptlist[:,0,:])**2)
		residual2 = np.sum((project(transform(R, t, pts3)) - ptlist[:,1,:])**2)
		#residual = map(lambda pointnum, imagelabel, position: abserror(project(R[imagelabel]* ptposition3[ptnum] + t[imagelabel]) - position) , ptlist)
		return residual1 + residual2
	return bundle_error






cam = MyCam()
frame1 = cam.read()
pnts1 = cam.projectPoints()
mask1 = cam.mask()

angle = .1
rotateX = np.array([[1,0,0],
					[0,np.cos(angle), np.sin(angle)],
					[0,-np.sin(angle), np.cos(angle)]])

cam.R = np.dot(rotateX, cam.R)

frame2 = cam.read()
pnts2 = cam.projectPoints()
mask2 = cam.mask()

mask = np.logical_and(mask1, mask2)
pntlist = np.array([pnts1[mask], pnts2[mask]])


R = rotateX#np.identity(3)
t = np.zeros(3)
#pts3 = np.ones((pntlist.shape[0],3))
pts3 = np.array(cam.pointCloud)[mask,:]
x0 = pack(R,t,pts3)

resid = error_func(pntlist) #ptlist.shape = [:, 2, 2], [pt, image, position]
grad, hess = gh(resid)



def cb(x):
	print "step"
res = optimize.minimize(resid, x0, method='Newton-CG', jac=grad, hess=hess, callback=cb)

print np.sqrt(res.fun) / pntlist.shape[0]/2

R,t,pts3 = unpack(res.x)

print rotateX - R

'''
def abserror(err):
	return min(err**2, 100)

def bundle_error(ptlist):
	residual = map(lambda pointnum, imagelabel, position: abserror(project(R[imagelabel]* ptposition3[ptnum] + t[imagelabel]) - position) , ptlist)
	return residual
	#residual = min(np.abs(project(R[ptlist[:,1]] * position3[pointnum] + t[imagelabel]) - position), 1)
	#jacobian = 
'''

'''
def error_func(ptlist):
	def bundle_error(R1, R2, t1, t2, pts3):
		residual1 = np.sum(np.linalg.norm(project(transform(R1, t1, pts3) - ptlist[:,0,:], axis = 1))
		residual2 = np.sum(np.linalg.norm(project(transform(R2, t2, pts3) - ptlist[:,1,:], axis = 1)) 
		#residual = map(lambda pointnum, imagelabel, position: abserror(project(R[imagelabel]* ptposition3[ptnum] + t[imagelabel]) - position) , ptlist)
		return residual1 + residual2
	return bundle_error
'''


'''
(pointnum, imagelabel, position)
(position1, position2, position3, ..)

ptlist[0,0]=pointnum
'''




