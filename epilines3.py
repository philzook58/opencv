#from matplotlib import pyplot as plt
import numpy as np

import cv2



#img1 = cv2.imread('left.jpg')  #queryimage # left image
#img2 = cv2.imread('right.jpg') #trainimage # right image

sift = cv2.SIFT()
cap = cv2.VideoCapture(0)

# find the keypoints and descriptors with SIFT
_, frame = cap.read()
_, frame = cap.read()
frame = cv2.pyrDown(frame)
kp2, des2 = sift.detectAndCompute(frame,None)
img2 = frame
rotation = np.identity(3)
h = frame.shape[0]
w = frame.shape[1]


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


while True:

    _, frame = cap.read()
    frame = cv2.pyrDown(frame)

    img1 = frame
    kp1, des1 = sift.detectAndCompute(img1,None)
    #kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    #print pts2



    for pt1 in pts2:
        cv2.circle(img1,tuple(pt1),5,[0,0,255],-1)

    def normPts(pts):
        pts[:,0] = pts[:,0]/w - 0.5 
        pts[:,1] = pts[:,1]/h - 0.5
        return pts

    normpts1 = normPts(pts1)
    normpts2 = normPts(pts2)

    F, mask = cv2.findFundamentalMat(normpts1,normpts2,cv2.FM_RANSAC)


    pts1 = pts1[mask.ravel()==1]



    #cv2.imshow('img3',img1)
    #cv2.waitKey(5)
    #print F
    U, s, V = np.linalg.svd(F) #, full_matrices=True) #Note that V here is often called V.H eslewhere in literature
    S = np.diag(s) #sorted in descending order
    W = np.array([[0, 1, 0],
    			  [-1,0, 0],
    			  [0, 0, 1]])
    R1 = np.dot( np.dot(U, W.T) , V)
    R2 = np.dot( np.dot(U, W) , V)
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    if np.linalg.norm(R1- np.identity(3)) <  np.linalg.norm( R2 - np.identity(3)) :
        R = R1
    else: 
        R = R2
    #print np.linalg.det(R)
    #if np.linalg.det(R) < 0:
    #    R = -R
    #print np.dot(R, R.T)
    T = np.dot(F, np.linalg.inv(R))
    t = np.array([T[1,2], T[2,0], T[0,1]])

    #if s[1] > s[0]/2:
    #    rotation = np.dot(rotation, R)
    rotation = np.dot(R, rotation)

    cv2.imshow('frame',drawRotate(rotation,img1))


    kp2, des2  = kp1, des1 
    img2 = img1

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()





