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

def imhist(im):
  # calculates normalized histogram of an image
	h = [0.0] * 256
	for i in range(3):
		for j in range(3):
			h[im[i][j]]+=1
	return np.array(h)/(3*3)

def cumsum(h):
	# finds cumulative sum of a numpy array, list
	return [sum(h[:i+1]) for i in range(len(h))]

def histeq(im):
	#calculate Histogram
	h = imhist(im)
	cdf = np.array(cumsum(h)) #cumulative distribution function
	sk = np.uint8(255 * cdf) #finding transfer function values
	s1, s2 = 3,3
	Y = np.zeros_like(im)
	# applying transfered values for each pixels
	for i in range(0, s1):
		for j in range(0, s2):
			Y[i][j] = sk[im[i][j]]
	H = imhist(Y)
	#return transformed image, original and new istogram, 
	# and transform function
	return Y , h, H, sk
# Adabtive equalization
def main(imgName):
	#Load image
	#Grey scale
	gray_img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE) 
	#Returns the dimentions of each color and number of cololers 
	w,h = gray_img.shape
	#Bordered image new size 
	bw = w+2
	bh = h+2 
	# set bordars to the image
	boarded_graypixels = [[0 for x in range(bh)] for y in range(bw)] 
	# Adabted image
	adabted_img = [[1 for x in range(h)] for y in range(w)] 
	#Load Image pixels in List Of lists
	for i in range(w):
		for j in range(h):
			boarded_graypixels[i+2][j+2] = gray_img[i,j]
	
	# take sub window 3*3
	window = [[0 for x in range(3)] for y in range(3)] 
	# Loop on the bordered image
	x,y=2,2
	for x in range(bw-2):
		for y in range(bh-2):
			# loop to fill current window
			for k in range(3):
				for l in range(3):
					window[k][l] = boarded_graypixels[x+k][y+l]

			equ_window, h, new_h, sk = histeq(window)
			#Set Pixel equalized at the senter to adabted image
			adabted_img[x][y] = equ_window[1][1]
			print adabted_img[x][y]
	print adabted_img
	Adapted_hist = manual_histogram(adabted_img ,w,h)
	#Adapted_hist = imhist(adabted_img)

	res = np.hstack((gray_img,adabted_img)) #stacking images side-by-side 
	cv2.imwrite('Gray-Man-Addaptive2.jpg',res)

	print Adapted_hist
	plt.plot(Adapted_hist,color = 'gray')
	plt.xlim([0,256])
	plt.ylim([0,800])
	plt.show()

	# End Loop on image
	#full_adabted_Img = np.hstack((gray_img,adabted_img)) #stacking images side-by-side 
	#cv2.imwrite('Adapted_Result.png',full_adabted_Img)


main('space.jpg')
