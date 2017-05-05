# CS6476 Fall 2015
# Problem Set 1: Images as Functions
# Luke Wicent Sy

import numpy as np
import cv2

def get_stats(img1):
    ret = { 'min': img1.min(),
            'max': img1.max(), 
            'mean': img1.mean(),
            'std': img1.std() }
    return ret

# 1 Input Images
# since the images in the database are all squares, I transformed them into rectangles manually
img_in1 = cv2.imread('input/4.2.04.tiff', 1)
img_in2 = cv2.imread('input/4.2.06.tiff', 1)

img1_bgr = img_in1[:, 57:425]
img2_bgr = img_in2[225:,:]
cv2.imwrite('output/ps1-1-a-1.png', img1_bgr)
cv2.imwrite('output/ps1-1-a-2.png', img2_bgr)

# 2 Color Planes
img1_bgr2 = np.array(img1_bgr)
img1_bgr2[:,:,0] = img1_bgr[:,:,2] #blue <- red
img1_bgr2[:,:,2] = img1_bgr[:,:,0] #red <- blue
cv2.imwrite('output/ps1-2-a-1.png', img1_bgr2)

img1_green = img1_bgr[:,:,1]
cv2.imwrite('output/ps1-2-b-1.png', img1_green)
img1_red = img1_bgr[:,:,2]
cv2.imwrite('output/ps1-2-c-1.png', img1_red)

# 3 Replacement of pixels
img1_gray = np.array(img1_green)
img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
yStart1, yEnd1 = img1_gray.shape[0]/2 - 50, img1_gray.shape[0]/2 + 50
xStart1, xEnd1 = img1_gray.shape[1]/2 - 50, img1_gray.shape[1]/2 + 50
img1_cut = img1_gray[yStart1:yEnd1, xStart1:xEnd1]
yStart2, yEnd2 = img2_gray.shape[0]/2 - 50, img2_gray.shape[0]/2 + 50
xStart2, xEnd2 = img2_gray.shape[1]/2 - 50, img2_gray.shape[1]/2 + 50
img2_gray[yStart2:yEnd2, xStart2:xEnd2] = img1_cut
cv2.imwrite('output/ps1-3-a-1.png', img2_gray)

# 4 Arithmetic and Geometric Operations
img1_gstats = get_stats(img1_green)
print "image1_green min: {min} max: {max} mean: {mean:0.2f} std: {std:0.2f}".format(**img1_gstats)

img1_green2 = ((img1_green - img1_gstats['mean']) / img1_gstats['std'] * 10) + img1_gstats['mean']
cv2.imwrite('output/ps1-4-b-1.png', img1_green2)

img1_green3 = np.zeros(img1_green.shape)
img1_green3[:,:-2] = img1_green[:,2:]
cv2.imwrite('output/ps1-4-c-1.png', img1_green3)

img1_green_diff = img1_green - img1_green3
print "image1_green_diff min: {min} max: {max} mean: {mean:0.2f} std: {std:0.2f}".format(**get_stats(img1_green_diff))
cv2.normalize(img1_green_diff,img1_green_diff,0,255,cv2.NORM_MINMAX)
cv2.imwrite('output/ps1-4-d-1.png', img1_green_diff)

# 5 Noise
sigma = 3
np.random.seed(125)
img1_noise1 = np.array(img1_bgr)
img1_noise1[:,:,1] += np.random.normal(loc=0,scale=sigma,size=img1_bgr.shape[:2])
cv2.imwrite('output/ps1-5-a-1.png', img1_noise1)

img1_noise2 = np.array(img1_bgr)
img1_noise2[:,:,0] += np.random.normal(loc=0,scale=sigma,size=img1_bgr.shape[:2])
cv2.imwrite('output/ps1-5-b-1.png', img1_noise2)
