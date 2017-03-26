from scipy.io import savemat
import matplotlib.pyplot as plt
import numpy as np

def edge_sens_interp_1(raw_img):
    raw_shape = raw_img.shape
    #plt.axis('off')
    #plt.imshow(raw_img, cmap='gray')
    #plt.show() 
    
    new_shape = tuple( 2*i for i in raw_shape)
    new_img = np.zeros(new_shape, dtype=np.float64)
    
    # Step 1: interpolate the missing pixels along the diagonal
    for i in range(raw_shape[0] - 1):
        for j in range(raw_shape[1] - 1):
            top_left = raw_img[i, j]
            top_right = raw_img[i, j+1]
            bottom_left = raw_img[i+1, j]
            bottom_right = raw_img[i+1, j+1]
            right_diag = np.abs(top_right - bottom_left)
            left_diag = np.abs(top_left - bottom_right)        
            diag_pix_idx = (2*i + 1, 2*j + 1)
            new_img[2*i, 2*j] = raw_img[i, j] # Copy old image values
            if left_diag < right_diag:
                new_img[diag_pix_idx] = (top_left + bottom_right)/ 2
            else:
                new_img[diag_pix_idx] = (top_right + bottom_left)/ 2
    
    # Step 2: interpolate the other half missing pixels
    for i in range(1, new_shape[0] - 1):
        for j in range(1, raw_shape[1] - 1):
            if i%2 == 1:
                top = new_img[i-1, 2*j]
                right = new_img[i, 2*j+1]
                left = new_img[i, 2*j-1]
                bottom = new_img[i+1, 2*j]
                center_pix_idx = (i, 2*j)
            else:
                top = new_img[i-1, 2*j-1]
                right = new_img[i, 2*j]
                left = new_img[i, 2*j-2]
                bottom = new_img[i+1, 2*j-1]
                center_pix_idx = (i, 2*j-1)
            horizontal_diff = np.abs(left-right)
            vertical_diff = np.abs(top-bottom)
            if horizontal_diff < vertical_diff:
                new_img[center_pix_idx] = (left + right)/2
            else:
                new_img[center_pix_idx] = (top + bottom)/2
    return new_img

# #new_img /= 255
# total_scale = 4
# scale = total_scale
# filename = 'camera.tif'
# raw_img = plt.imread(filename).astype(np.float64)
# new_img = edge_sens_interp_1(raw_img)
# while( scale/ 2 != 1):
    # new_img = edge_sens_interp_1(new_img)
    # scale /= 2
# matlab_mat = {'data': new_img}
# savemat('newim1.mat', matlab_mat)
# plt.imsave('new1.jpg', new_img, cmap='gray') #, vmin=0, vmax=255
# print('Image 1 finished')

# scale = total_scale
# filename = 'ic.tif'
# raw_img = plt.imread(filename).astype(np.float64)
# new_img = edge_sens_interp_1(raw_img)
# while( scale/ 2 != 1):
    # new_img = edge_sens_interp_1(new_img)
    # scale /= 2
# matlab_mat = {'data': new_img}
# savemat('newim2.mat', matlab_mat)
# plt.imsave('new2.jpg', new_img, cmap='gray')  #, vmin=0, vmax=255
# print('Image 2 finished')
#plt.axis('off')
#plt.imshow(new_img, cmap='gray', interpolation='none')

color = ['airplane.png', 'baboon.png', 'fruits.png', 'frymire.png', 'lena.png', 'peppers.png']
gray = ['barbara.png', 'bike.png', 'boat.png', 'fprint3.png', 'goldhill.png', 'zelda.png']

in_dir = '128'
out_dir = 'out2'
total_scale = 4 
for filename in gray:
# scale = total_scale
# filename = 'camera.tif'
# raw_img = plt.imread(filename).astype(np.float64)
# new_img = edge_sens_interp_1(raw_img)
# while( scale/ 2 != 1):
    # new_img = edge_sens_interp_1(new_img)
    # scale /= 2
# matlab_mat = {'data': new_img}
# savemat('newim1.mat', matlab_mat)
# plt.imsave('new1.jpg', new_img, cmap='gray') #, vmin=0, vmax=255
# print('Image 1 finished')
    scale = total_scale
    raw_img = plt.imread(in_dir + '/' + filename).astype(np.float64)
    new_img_1 = edge_sens_interp_1(raw_img[:,:,0])
    new_img_2 = edge_sens_interp_1(raw_img[:,:,1])
    new_img_3 = edge_sens_interp_1(raw_img[:,:,2])
    new_img = np.dstack((new_img_1, new_img_2, new_img_3))
    while( scale/ 2 != 1):
        new_img_1 = edge_sens_interp_1(new_img[:,:,0])
        new_img_2 = edge_sens_interp_1(new_img[:,:,1])
        new_img_3 = edge_sens_interp_1(new_img[:,:,2])
        new_img = np.dstack((new_img_1, new_img_2, new_img_3))
        scale /= 2
    matlab_mat = {'data': new_img}
    #savemat('newim2.mat', matlab_mat)
    plt.imsave(out_dir + '/' + filename[:-4] + '_edge' + filename[-4:], new_img)
    print filename
    #plt.axis('off')
    #plt.imshow(new_img, cmap='gray', interpolation='none')
    
for filename in color:
    scale = total_scale
    raw_img = plt.imread(in_dir + '/' + filename).astype(np.float64)
    new_img_1 = edge_sens_interp_1(raw_img[:,:,0])
    new_img_2 = edge_sens_interp_1(raw_img[:,:,1])
    new_img_3 = edge_sens_interp_1(raw_img[:,:,2])
    new_img = np.dstack((new_img_1, new_img_2, new_img_3))
    while( scale/ 2 != 1):
        new_img_1 = edge_sens_interp_1(new_img[:,:,0])
        new_img_2 = edge_sens_interp_1(new_img[:,:,1])
        new_img_3 = edge_sens_interp_1(new_img[:,:,2])
        new_img = np.dstack((new_img_1, new_img_2, new_img_3))
        scale /= 2
    matlab_mat = {'data': new_img}
    #savemat('newim2.mat', matlab_mat)
    plt.imsave(out_dir + '/' + filename[:-4] + '_edge' + filename[-4:], new_img)
    print filename
    #plt.axis('off')
    #plt.imshow(new_img, cmap='gray', interpolation='none')
