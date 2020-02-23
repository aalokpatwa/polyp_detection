'''
unet_polyp uses elastic_deformations on the dataset to augment the images before the training begins.
training will use other image augmentation techniques available in Keras

'''
# Affine Transform and Elastic Deformation
# Affine Transform: translation, rotation, skewing
# First find a displacement vector, then for every pixel, replace it with pixel at displacement vector
# if displacement vector is outside the image, it is 0
# pixel at displacement vector is calculated with bilinear interpolation. 
# bilinear interpolation: first do horizontal interpolation, then do verticle. 
# Do this for each channel. 
# Affine Transform can be commented and uncommented
# Elastic Deformation: 
# Degree of transformation can be controlled by parameters
# To plot, uncomment matplot related lines

basedir = "/unet_polyp/"
augcnt = 20
DBDIR = basedir+"datasets/cvc300/"
IMDIR = DBDIR+ "images/"
MASKDIR = DBDIR + "gtpolyp/"
AUGIMDIR = DBDIR + "augimages/"
AUGMASKDIR = DBDIR + "auggtpolyp/"


# Import stuff
import os
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')



def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2  # this is Floor integer division. Module is %
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    if (alpha_affine != 0):
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    
    # Elastic Deformation
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

    

#make output directories
if not os.path.exists(AUGIMDIR):
    os.mkdir(AUGIMDIR)
if not os.path.exists(AUGMASKDIR):
    os.mkdir(AUGMASKDIR)


for filename in os.listdir(IMDIR):
    filename_wo_ext = filename.split(".")[0]
    print(filename_wo_ext)
    image_name = os.path.join(IMDIR+filename)
    mask_name = os.path.join(MASKDIR+filename)
    #OpenCV represents RGB images in reverse order! BGR rather than RGB
    #So to plot it, we need to convert it back
    im = cv2.imread(image_name, cv2.IMREAD_COLOR)
    im_mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
    print(im.shape, im_mask.shape)
    im_mask_reshaped = np.reshape(im_mask, (im_mask.shape[0], im_mask.shape[1], 1))  #to concatenate

# Display result
#print(im.shape, im_mask.shape, im_mask_reshape.shape)
#plt.figure(figsize = (16,14))
#plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
#plt.figure(figsize = (16,14))
#plt.imshow(im_mask)

    # Merge images into separete channels (shape will be (rows, columns, 4))
    im_merge = np.concatenate((im, im_mask_reshaped), axis=2)

    for augno in range(augcnt):
    # elastic transformation parameters: alpha, sigma, alpha_affine
        alpha = 4
        sigma = 0.05
        alpha_affine = 0
    # Apply transformation on image
        im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * alpha, im_merge.shape[1] * sigma, im_merge.shape[1] * alpha_affine)
    # Split image and mask
        b,g,r,m = np.split(im_merge_t, 4, 2)
        im_t = np.concatenate((b,g,r), axis=2)
        im_mask_t_reshaped = m
        im_mask_t = np.reshape(im_mask_t_reshaped, (im_mask_t_reshaped.shape[0], im_mask_t_reshaped.shape[1])) # go back to original
        cv2.imwrite(AUGIMDIR+filename.split(".")[0]+"_aug"+str(augno+1)+".png", im_t)
        cv2.imwrite(AUGMASKDIR+filename.split(".")[0]+"_aug"+str(augno+1)+".png", im_mask_t)
    # write original images as _aug0
    cv2.imwrite(AUGIMDIR+filename.split(".")[0]+"_aug"+str(0)+".png", im)
    cv2.imwrite(AUGMASKDIR+filename.split(".")[0]+"_aug"+str(0)+".png", im_mask)

# Display result
#print(b.shape, g.shape, r.shape, im_t.shape, m.shape, im_mask_t_back.shape)
#plt.figure(figsize = (16,14))
#plt.imshow(cv2.cvtColor(im_t, cv2.COLOR_BGR2RGB))
#plt.figure(figsize = (16,14))
#plt.imshow(im_mask_t_back)

