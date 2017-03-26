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
def plot_Histogram(bgr):
	#plot	
	color = ('b','g','r')
	for i,col in enumerate(color):
	    plt.plot(bgr[i],color = col)
	    plt.xlim([0,260])
	    plt.ylim([0,800])
	plt.show()



def main(cimgName):
	# RGB image
	c_img = cv2.imread(cimgName, cv2.IMREAD_COLOR)
	# coloered Image
	#Returns the dimentions of each color and number of cololers 
	x,y,z = c_img.shape	
	r_Pixels = [[0 for f in range(y)] for k in range(x)] 
	g_Pixels = [[0 for f in range(y)] for k in range(x)] 
	b_Pixels = [[0 for f in range(y)] for k in range(x)] 
	#Load pixels in List Of lists
	for i in range(x):
		for j in range(y):
			r_Pixels[i][j] = c_img[i,j,0]
			g_Pixels[i][j] = c_img[i,j,1]
			b_Pixels[i][j] = c_img[i,j,2]
	# Calculat RGB Histo
	red_histogram = manual_histogram(r_Pixels,x,y)
	green_histogram = manual_histogram(g_Pixels,x,y)
	blue_histogram = manual_histogram(b_Pixels,x,y)

	BGR = [red_histogram , green_histogram , blue_histogram]
	plot_Histogram(BGR)

main('tiger.jpg')
