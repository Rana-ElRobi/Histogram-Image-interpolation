# ready_equ_Gray.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pylab as pl

def main(imgName):
	# Ready one 
	# HElper Link 
	# http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
	#Grey scale
	gray_img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
	equalized_image = cv2.equalizeHist(gray_img)
	res = np.hstack((gray_img,equalized_image)) #stacking images side-by-side 
	cv2.imwrite('equalizedImage.png',res)
	#check histogram updates
	hist = cv2.calcHist([equalized_image],[0],None,[256],[0,256])
	plt.plot(hist,color = 'gray')
	plt.xlim([0,256])
	plt.ylim([0,800])
	plt.show()


main('stones.jpg')
#main('space.jpg')