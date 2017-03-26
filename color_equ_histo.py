import pylab as plt
import matplotlib.image as mpimg
import numpy as np 
import cv2



def imhist(im):
  # calculates normalized histogram of an image
	m, n = im.shape
	h = [0.0] * 256
	for i in range(m):
		for j in range(n):
			h[im[i, j]]+=1
	return np.array(h)/(m*n)

def cumsum(h):
	# finds cumulative sum of a numpy array, list
	return [sum(h[:i+1]) for i in range(len(h))]

def histeq(im):
	#calculate Histogram
	h = imhist(im)
	cdf = np.array(cumsum(h)) #cumulative distribution function
	sk = np.uint8(255 * cdf) #finding transfer function values
	s1, s2 ,z = im.shape
	Y = np.zeros_like(im)
	# applying transfered values for each pixels
	for i in range(0, s1):
		for j in range(0, s2):
			Y[i, j] = sk[im[i, j]]
	H = imhist(Y)
	#return transformed image, original and new istogram, 
	# and transform function
	return Y , h, H, sk

#function plots histogram for gray image
def plot_Histogram(bgr):
	#plot	
	color = ('b','g','r')
	for i,col in enumerate(color):
	    plt.plot(bgr[i],color = col)
	    plt.xlim([0,256])
	    plt.ylim([0,1600])
	plt.show()

def main (imgName):
	# load image to numpy arrayb
	# matplotlib 1.3.1 only supports png images
	# use scipy or PIL for other formats
	img = np.uint8(mpimg.imread(imgName)*255.0)
	new_img, h, new_h, sk = histeq(img)
	# coloered Image
	#Returns the dimentions of each color and number of cololers 
	x,y,z = c_img.shape	
	r_Pixels = [[0 for f in range(y)] for k in range(x)] 
	g_Pixels = [[0 for f in range(y)] for k in range(x)] 
	b_Pixels = [[0 for f in range(y)] for k in range(x)] 
	#Load pixels in List Of lists
	for i in range(x):
		for j in range(y):
			b_Pixels[i][j] = c_img[i,j,0]
			g_Pixels[i][j] = c_img[i,j,1]
			r_Pixels[i][j] = c_img[i,j,2]
	# Calculat RGB Histo
	red_histogram = imhist(r_Pixels,x,y)
	green_histogram = imhist(g_Pixels,x,y)
	blue_histogram = imhist(b_Pixels,x,y)

	bgr_before = [blue_histogram , green_histogram,red_histogram  ]
	plot_Histogram(bgr_before)

	Requ_pixels, r_h, r_new_h, r_sk = histeq(r_Pixels,x,y)
	Gequ_pixels, g_h, g_new_h, g_sk = histeq(g_Pixels,x,y)
	Bequ_pixels, b_h, b_new_h, b_sk = histeq(b_Pixels,x,y)

	Requ_histo = imhist(Requ_pixels,s1,s2)
	Gequ_histo = imhist(Gequ_pixels,s1,s2)
	Bequ_histo = imhist(Bequ_pixels,s1,s2)
	
	BGR = [Bequ_histo , Gequ_histo , Requ_histo]

	plot_Histogram(BGR)
	# # show old and new image
	# # show original image
	# plt.subplot(121)
	# plt.imshow(img)
	# plt.title('original image')
	# plt.set_cmap('gray')
	# # show original image
	# plt.subplot(122)
	# plt.imshow(new_img)
	# plt.title('hist. equalized image')
	# plt.set_cmap('gray')
	# plt.show()
	# # plot histograms and transfer function
	# fig = plt.figure()
	# fig.add_subplot(221)
	# plt.plot(h)
	# plt.title('Original histogram') # original histogram
	# fig.add_subplot(222)
	# plt.plot(new_h)
	# plt.title('New histogram') #hist of eqlauized image
	# fig.add_subplot(223)
	# plt.plot(sk)
	# plt.title('Transfer function') #transfer function
	# plt.show()


main('tiger.jpg')
