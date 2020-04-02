# Author: Moumita Paul
# Project3 : Buoy Detection


import cv2 
import numpy as np 
from matplotlib import pyplot as plt
import math
from matplotlib.lines import Line2D
from numpy.random import rand


# Splitting given video into frames
cap = cv2.VideoCapture('detectbuoy.avi')
while(cap.isOpened()):
	ret, frame = cap.read()
	if ret == True:
		cv2.waitKey(0)
		cv2.imwrite('1st_Frame.jpg',frame)
		cv2.destroyAllWindows()
		break

# 
crop = False
copy_frame = frame.copy()
x_init,y_init, x_final, y_final = 0,0,0,0

# Click function
def onclick(event,x,y,flags,param):
 	global x_init, y_init, x_final, y_final, crop

 	if event ==cv2.EVENT_LBUTTONDOWN:
 		x_init,y_init, x_final, y_final = x, y, x, y
 		crop = True
 	elif event == cv2.EVENT_MOUSEMOVE:
 		if crop == True:
 			x_final,y_final = x,y

 	elif event == cv2.EVENT_LBUTTONUP:
 		x_final,y_final = x,y
 		crop = False
 		ref_point = [(x_init, y_init), (x_final, y_final)]

 		# Two points found
 		if len(ref_point) == 2: 
		    roi = copy_frame[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
		    cv2.imshow("Cropped", roi)
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", onclick)
while True:
 
    i = frame.copy()
 
    if not crop:
        cv2.imshow("frame", frame)
 
    elif crop:
        cv2.rectangle(i, (x_init, y_init), (x_final, y_final), (255, 0, 0), 2)
        cv2.imshow("frame", i)
 
    cv2.waitKey(1)
 
# close all open windows
cv2.destroyAllWindows()


