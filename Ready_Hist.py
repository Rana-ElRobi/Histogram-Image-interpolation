# Ready_Histogram
# Helper Link
# http://docs.opencv.org/trunk/d1/db7/tutorial_py_histogram_begins.html
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pylab as pl

def main(imgName):
	img = cv2.imread(imgName,0)
	hist = cv2.calcHist([img],[0],None,[256],[0,256])
	plt.plot(hist,color = 'gray')
	plt.xlim([0,256])
	plt.ylim([0,800])
	plt.show()


main('stones.jpg')

#main('space.jpg')