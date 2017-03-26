import numpy as np
import cv2
import matplotlib.pyplot as plt
import pylab as pl

def manual_histogram(pixels_List,width,hight):
	#initiate list of counters each cell represent intensty
	intenisty_frequency = [0]*256
	for i in range (width):
		for j in range (hight):
			#counter ++ of he current pixel intenisty
			intenisty_frequency[pixels_List[i][j]] += 1
	return intenisty_frequency

#function plots histogram for gray image
def plot_Histogram(grayhisto):
	#plot	
	plt.plot(grayhisto,color = 'gray')
	plt.xlim([0,256])
	plt.ylim([0,1600])
	plt.show()

def main(imgName):
	img = cv2.imread(imgName,0)
	# create a CLAHE object (Arguments are optional).
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(img)
	res = np.hstack((img,cl1)) #stacking images side-by-side 
	#cv2.imwrite('equalizedImage.png',res)
	cv2.imwrite('readyAddaptive2.jpg',res)
	x,y = cl1.shape
	# Draw histogram for updated img
	adapt_hist = manual_histogram(cl1 ,x,y )
	plot_Histogram(adapt_hist)
	#cv2.imshow(cl1)

main('stones.jpg')
#main('space.jpg')
