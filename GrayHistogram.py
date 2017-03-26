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
	plt.ylim([0,800])
	plt.show()

def main(imgName):
	#Grey scale
	gray_img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE) 
	#Returns the dimentions of each color and number of cololers 
	w,h = gray_img.shape
	# GreyScale Image
	gray_Pixels= [[0 for x in range(h)] for y in range(w)] 
	#Load pixels in List Of lists
	for i in range(w):
		for j in range(h):
			gray_Pixels[i][j] = gray_img[i,j]
	# Calculate Histogram	
	gray_histo = manual_histogram(gray_Pixels ,w,h)
	#plot gray_histogram
	plot_Histogram(gray_histo)

main('stones.jpg')

#main('space.jpg')
