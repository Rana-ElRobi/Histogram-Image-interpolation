from scipy.io import savemat
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.misc import imresize
from scipy.signal import fftconvolve
import os

def nearest_kernel(scale = 2):
    k_size = scale*2 + 1
    k = np.zeros(k_size)
    for i in range(k_size):
        x = ( (i-1.*k_size)/k_size)
        x_abs = abs(x)
        if x_abs <= 0.5:
            k[i] = 1
    return k

def lanczos_kernel(scale = 2):
    a = 2.0
    k_size = scale*2*2 + 1
    k = np.zeros(k_size)
    for i in range(k_size):
        x = ( (i-2.*scale)/scale)
        x_abs = abs(x)
        if x_abs < a:
            k[i] = np.sinc(x_abs)*np.sinc(x_abs/a)
    return k

def bilinear_kernel(scale = 2):
    k_size = scale*2 + 1
    k = np.zeros(k_size)
    for i in range(k_size):
        x = ( (i-1.*scale)/scale)
        x_abs = abs(x)
        k[i] = 1 - x_abs
    return k
    
def bell_kernel(scale = 2):
    k_size = scale*2 + 1
    k = np.zeros(k_size)
    for i in range(k_size):
        x = ( (i-1.*scale)/scale )
        x_abs = abs(x)
        if x_abs < 0.5:
            k[i] = 0.75 - x_abs**2
        elif x_abs < 1.5:
            k[i] = 0.5*((x_abs-1.5))**2
    return k

def bicubic_kernel(scale = 2):
    k_size = scale*2*2 + 1
    k = np.zeros(k_size)
    a = -0.6
    for i in range(k_size):
        x = ( (i-2.*scale)/scale)
        x_abs = abs(x)
        if x_abs <= 1:
            k[i] = (a+2)*x_abs**3 - (a+3)*(x_abs**2) + 1
        elif x_abs <= 2:
            k[i] = a*(x_abs**3) - (5*a)*(x_abs**2) + 8*a*x_abs - 4*a
    return k
    
def hermite_kernel(scale = 2):
    k_size = scale*2*2 + 1
    k = np.zeros(k_size)
    a = 0
    for i in range(k_size):
        x = ( (i-2.*scale)/scale)
        x_abs = abs(x)
        if x_abs <= 1:
            k[i] = (a+2)*x_abs**3 - (a+3)*(x_abs**2) + 1
        elif x_abs <= 2:
            k[i] = a*(x_abs**3) - (5*a)*(x_abs**2) + 8*a*x_abs - 4*a
    return k
    
def interpolate(im, k, scale=2):
    img = np.zeros((im.shape[0]*scale, im.shape[1]*scale))
    for i in range(0, img.shape[0]-scale, scale):
        for j in range(0, img.shape[1]-scale, scale):
            if (i/scale + scale - 1) < im.shape[0] and (j/scale + scale - 1) < im.shape[1]:
                img[i + scale - 1][j + scale - 1] = im[i/scale + scale - 1][j/scale + scale - 1]
    k = k.reshape((-1,1))
    convolved_img = fftconvolve(img, k.T*k, 'same')
    return convolved_img

def interpolateRGB(im, k, scale=2):
    img = np.zeros((im.shape[0]*scale, im.shape[1]*scale,3))
    for i in range(0, img.shape[0]-scale, scale):
        for j in range(0, img.shape[1]-scale, scale):
            if (i/scale + scale - 1) < im.shape[0] and (j/scale + scale - 1) < im.shape[1]:
                img[i + scale - 1][j + scale - 1][0] = im[i/scale + scale - 1][j/scale + scale - 1][0]
                img[i + scale - 1][j + scale - 1][1] = im[i/scale + scale - 1][j/scale + scale - 1][1]
                img[i + scale - 1][j + scale - 1][2] = im[i/scale + scale - 1][j/scale + scale - 1][2]
    k = k.reshape((-1,1))
    convolved_img = np.zeros((im.shape[0]*scale, im.shape[1]*scale,3))
    convolved_img[:,:,0] = fftconvolve(img[:,:,0], k.T*k, 'same')
    convolved_img[:,:,1] = fftconvolve(img[:,:,1], k.T*k, 'same')
    convolved_img[:,:,2] = fftconvolve(img[:,:,2], k.T*k, 'same')
    return convolved_img    
    
    
def convolve(img, kernel, scale=1):
    #find center position of kernel (half of kernel size)
    kCols = kernel.shape[0]
    rows = img.shape[0]
    cols = img.shape[1]
    kCenterX = kernel.shape[0] / 2
    kCenterY = kernel.shape[0] / 2
    out = np.zeros((rows, cols))
    tmp_out = np.zeros((rows, cols))
    for i in range(scale-1, rows, scale): #rows
        for j in range(scale-1, cols): #columns
            for m in range(kCols): #kernel rows
                imgConvIndex = m - 2.*scale
                colIndex = j + imgConvIndex
                rowIndex = i
                if rowIndex >= 0 and rowIndex < rows and colIndex >= 0 and colIndex < cols:
                    tmp_out[i][j] += img[rowIndex][colIndex]*kernel[m] 

    for i in range(scale, rows-1): #rows
        if i % scale == scale - 1:
            continue
        for j in range(cols): #columns
            tmpSum = 0
            for m in range(kCols): #kernel rows
                imgConvIndex = m - 2.*scale
                colIndex = j
                rowIndex = i  + imgConvIndex
                if rowIndex >= 0 and rowIndex < rows and colIndex >= 0 and colIndex < cols:
                    tmpSum += tmp_out[rowIndex][colIndex]*kernel[m]
            if rowIndex >= 0 and rowIndex < rows and colIndex >= 0 and colIndex < cols:
                out[i][j] = tmpSum
    return out + tmp_out

kernels = [bilinear_kernel, bicubic_kernel, lanczos_kernel, bell_kernel, hermite_kernel, nearest_kernel]
in_dir = '128'
out_dir = 'out2'
imnames = os.listdir(in_dir)

color = ['airplane.png', 'baboon.png', 'fruits.png', 'frymire.png', 'lena.png', 'peppers.png']
gray = ['barbara.png', 'bike.png', 'boat.png', 'fprint3.png', 'goldhill.png', 'zelda.png']

scale = 4

for fname in gray:
    print fname,
    imname = in_dir + '/' + fname
    image = Image.open(imname).convert('L')
    image = np.asarray(image, dtype=np.float64)
    img = np.zeros((image.shape[0]+2*scale, image.shape[1]+2*scale))
    img[scale:-scale, scale:-scale] = image
    for kernel in kernels:
        print kernel.func_name,
        kernel_2scale = kernel(scale)
        our_interp = interpolate(img, kernel_2scale, scale)
        our_interp = our_interp[scale:image.shape[0]*scale+scale, scale:image.shape[1]*scale+scale]
        plt.imsave(out_dir + '/' + fname[:-4] + '_' + kernel.func_name + fname[-4:], our_interp, cmap='gray', vmin=0, vmax=255)
    print ''
    
for fname in color:
    print fname,
    imname = in_dir + '/' + fname
    image = plt.imread(imname).astype(np.float64)[:,:,:3]
    image = np.asarray(image, dtype=np.float64)
    img = np.zeros((image.shape[0]+2*scale, image.shape[1]+2*scale, 3))
    img[scale:-scale, scale:-scale] = image
    for kernel in kernels:
        print kernel.func_name,
        kernel_2scale = kernel(scale)
        our_interp = interpolateRGB(img, kernel_2scale, scale)
        our_interp = our_interp[scale:image.shape[0]*scale+scale, scale:image.shape[1]*scale+scale]
        our_interp[our_interp < 0] = 0
        our_interp[our_interp > 1] = 1
        plt.imsave(out_dir + '/' + fname[:-4] + '_' + kernel.func_name + fname[-4:], our_interp)
    print ''
