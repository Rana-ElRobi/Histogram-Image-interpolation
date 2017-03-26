from scipy.io import savemat
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.misc import imresize
from scipy.signal import fftconvolve
import os

in_dir = '512'
out_dir = '128'
imnames = os.listdir(in_dir)

color = ['airplane.png', 'baboon.png', 'fruits.png', 'frymire.png', 'lena.png', 'peppers.png']
gray = ['barbara.png', 'bike.png', 'boat.png', 'fprint3.png', 'goldhill.png', 'zelda.png']

for fname in gray:
    print fname,
    imname = in_dir + '/' + fname
    image = Image.open(imname).convert('L')
    image = np.asarray(image, dtype=np.float64)
    our_interp = imresize(image, 0.25, 'bicubic')
    plt.imsave(out_dir + '/' + fname[:-4] + fname[-4:], our_interp, cmap='gray', vmin=0, vmax=255)
    print ''
    
for fname in color:
    print fname,
    imname = in_dir + '/' + fname
    image = plt.imread(imname).astype(np.float64)[:,:,:3]
    image = np.asarray(image, dtype=np.float64)
    our_interp = imresize(image, 0.25, 'bicubic')
    plt.imsave(out_dir + '/' + fname[:-4] + fname[-4:], our_interp, cmap='gray', vmin=0, vmax=255)
    print ''
