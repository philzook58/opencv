import cv2
import numpy as np

#cap = cv2.VideoCapture(0)

def draw(num):
    cv2.drawContours(frame, contours, num, (128,255,0), 3)
    cv2.imshow('res',cv2.pyrDown(frame))

cv2.namedWindow('image')
cv2.createTrackbar('Contour','image',0,10,draw)
#cv2.createTrackbar('Max Edge','image',200,255,nothing)

# Take each frame
#_, frame = cap.read()
frame = cv2.imread('myimage.png')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,100,200)

kernel = np.ones((5,5),np.uint8)
#erosion = cv2.erode(edges,kernel,iterations = 1)
edges = cv2.dilate(edges,kernel,iterations = 5) # really chunks it up
#ret2,edges = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#ret,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
#print cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]

contours,hierarchy= cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
#contours,hierarchy= cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[-2:]
#cv2.drawContours(frame, contours, -1, (128,255,0), 3)
cv2.drawContours(frame, contours, 5, (128,255,0), 3)

cnt = contours[0]
print cnt[0][0][0] # this is the format. contours is an array of double arrays. What the hell.
M = cv2.moments(cnt)
print M
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

cv2.imshow('res',cv2.pyrDown(frame))
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
