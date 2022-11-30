#!/usr/bin/env python3

# Python librairies
import numpy as np
import cv2
import cv2.aruco as aruco
from math import *
import time

n=3

img=cv2.imread("aruco_original.png", cv2.IMREAD_GRAYSCALE)

dim=(n*img.shape[0], n*img.shape[1])

img2 = np.ones(dim, np.uint8)
img2= 128*img2
img2[int((1-1/n)*dim[0]/2):int((1+1/n)*dim[0]/2), int((1-1/n)*dim[1]/2):int((1+1/n)*dim[1]/2)]=img

# cv2.imshow('aruco', img2)
# cv2.waitKey(0)
cv2.imwrite('aruco1.png', img2)