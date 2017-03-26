# Read_ColoredHistogram
# Helper Link
# http://docs.opencv.org/trunk/d1/db7/tutorial_py_histogram_begins.html

import cv2
import numpy as np
from matplotlib import pyplot as plt

def main(imgName):
	# RGB image
	img = cv2.imread(imgName, cv2.IMREAD_COLOR)#	
	color = ('b','g','r')
	for i,col in enumerate(color):
	    #histr = cv2.calcHist([img],[i],None,[256],[0,256])
	    hist = cv2.calcHist([img],[i],None,[256],[0,256])
	    plt.plot(hist,color = col)
	    plt.xlim([0,260])
	    plt.ylim([0,800])
	plt.show()


main('tiger.jpg')