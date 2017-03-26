import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

def main(imgName):
	img = cv2.imread(imgName)
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	# equalize the histogram of the Y channel
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
	# convert the YUV image back to RGB format
	img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
	# cv2.imshow('Color input image', img)
	# cv2.imshow('Histogram equalized', img_output)
	# cv2.waitKey(0)
	color = ('b','g','r')
	for i,col in enumerate(color):
	    #histr = cv2.calcHist([img],[i],None,[256],[0,256])
	    hist = cv2.calcHist([img_output],[i],None,[256],[0,256])
	    plt.plot(hist,color = col)
	    plt.xlim([0,256])
	    plt.ylim([0,500])
	plt.show()

main('tiger.jpg')
