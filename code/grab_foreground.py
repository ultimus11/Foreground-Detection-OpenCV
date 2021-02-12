import numpy as np
import cv2
import matplotlib.pyplot as plt

#read image
img = np.array(cv2.imread('1.jpg'))

#this is mask
mask = np.zeros(img.shape[:2],np.uint8)

#this bgdModel and fgdModel is used in background
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

#This is a rectangular cross section of given image where it will search for foreground
rect = (35,30,330,312)

#This is a grabcut func from opencv which is used to detect foreground
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

#here we show our image 
plt.imshow(img)
plt.colorbar()
plt.show()
cv2.imshow("sdfg",img)
cv2.waitKey(0)
cv2.imwrite("foreground.jpg",img)
