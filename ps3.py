"""Problem Set 3: Window-based Stereo Matching."""

import numpy as np
import cv2
from scipy import signal

import os
import sys

input_dir = "input"  # read images from os.path.join(input_dir, <filename>)
output_dir = "output"  # write images to os.path.join(output_dir, <filename>)

def custom_filter(img1, sigma=1.0, k=5):
    """ Convert pixel values that are sigma * std deviations above and below the median into the mean of the k x k window centered at the said pixel.
    
    Params:
    img1: input image
    sigma: scale of std
    k: Window Size (must be odd)

    Returns: filtered / normalized img1
    """
    img1_median = np.median(img1)
    upper_bound = img1_median + sigma * img1.std()
    lower_bound = img1_median - sigma * img1.std()
    kBorder = k>>1
    
    img2 = img1.copy()
    index = np.transpose(np.where((img1 > upper_bound) | (img1 < lower_bound)))

    for y, x in index:
        yStart = 0 if y == 0 else y - kBorder
        xStart = 0 if x == 0 else x - kBorder
        tmp = img1[yStart:y+kBorder+1,xStart:x+kBorder+1]
        tmp2 = tmp[np.where((tmp <= upper_bound) & (tmp >= lower_bound))]
        img2[y,x] = img1_median if tmp2.size == 0 else tmp2.mean()
        
    cv2.normalize(img2,img2,0,255,cv2.NORM_MINMAX)
    return img2
    
def custom_filter2(img1, sigma=1.0):
    """ Widen the pixel values that are within sigma * std deviations above and below the median, and Narrow the pixel values that are outside sigma * std deviations above and below the median.
    
    Params:
    img1: input image
    sigma: scale of std

    Returns: filtered / normalized img1
    """
    img1_median = np.median(img1)
    upper_bound = img1_median + sigma * img1.std()
    lower_bound = img1_median - sigma * img1.std()
    
    img2 = img1.copy()
    img2[img1 > upper_bound] = (img1[img1 > upper_bound] - upper_bound) / 10 + upper_bound
    img2[img1 < lower_bound] = img1[img1 < lower_bound] / 10 + (lower_bound * 9 / 10)
    cv2.normalize(img2,img2,0,255,cv2.NORM_MINMAX)
    return img2

def disparity_ssd(L, R, k=3):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image, in range [0.0, 1.0]
    R: Grayscale right image, same size as L
    k: Window Size (must be odd)

    Returns: Disparity map, same size as L, R
    """
    
    kBorder = k>>1
    D = np.zeros(L.shape, dtype=np.int16)
    L2 = cv2.copyMakeBorder(L,kBorder,kBorder,kBorder,kBorder,cv2.BORDER_CONSTANT,value=0)
    R2 = cv2.copyMakeBorder(R,kBorder,kBorder,kBorder,kBorder,cv2.BORDER_CONSTANT,value=0)
    DIFF2 = np.zeros((L2.shape[0],L2.shape[1],L2.shape[1]), dtype=L.dtype)
    
    for i in xrange(DIFF2.shape[2]):
        DIFF2[:,:DIFF2.shape[1]-i,i] = L2[:,:DIFF2.shape[1]-i] - R2[:,i:]
        DIFF2[:,DIFF2.shape[1]-i:,i] = L2[:,DIFF2.shape[1]-i:] - R2[:,:i]
    DIFF2 = DIFF2 * DIFF2  
    
    for y in xrange(L.shape[0]):
        for x in xrange(L.shape[1]):
            SSD = np.nansum(DIFF2[y:y+k,x:x+k,:], axis=(0,1))
            #SSD[L.shape[1]-x:L.shape[1]+k-x] = np.NaN
            x2 = (np.nanargmin(SSD)+x)%DIFF2.shape[1]
            D[y,x] = x2 - x

    return D


def disparity_ncorr(L, R, k=3):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image, in range [0.0, 1.0]
    R: Grayscale right image, same size as L
    k: Window Size (must be odd)

    Returns: Disparity map, same size as L, R
    """
    
    kBorder = k>>1
    D = np.zeros(L.shape, dtype=np.int16)
    L2 = cv2.copyMakeBorder(L,kBorder,kBorder,kBorder,kBorder,cv2.BORDER_CONSTANT,value=0)
    R2 = cv2.copyMakeBorder(R,kBorder,kBorder,kBorder,kBorder,cv2.BORDER_CONSTANT,value=0)
    
    for y in xrange(L.shape[0]):
        strip = R2[y:y+k,:]
        for x in xrange(L.shape[1]):
            #corr = signal.correlate2d(strip, L2[y:y+k,x:x+k], boundary='symm', mode='valid')
            corr = cv2.matchTemplate(strip, L2[y:y+k,x:x+k], cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)
            D[y,x] = max_loc[0] - x
            #D[y,x] = corr.argmax() - x
    return D


def main():
    """Run code/call functions to solve problems."""
    
    # 1-a
    # Read images
    L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0) * (1 / 255.0)  # grayscale, scale to [0.0, 1.0]
    R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0) * (1 / 255.0)
    k = 13
    
    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    D_L = disparity_ssd(L, R, k=k)
    D_R = disparity_ssd(R, L, k=k)
    print "pair0-D_L: shape={0}, min={1}, max={2}".format(D_L.shape, np.nanmin(D_L), np.nanmax(D_L))
    print "pair0-D_R: shape={0}, min={1}, max={2}".format(D_R.shape, np.nanmin(D_R), np.nanmax(D_R))
    
    # Save output images (D_L as output/ps3-1-a-1.png and D_R as output/ps3-1-a-2.png)
    # Note: They may need to be scaled/shifted before saving to show results properly
    cv2.normalize(D_L,D_L,0,255,cv2.NORM_MINMAX)
    cv2.normalize(D_R,D_R,0,255,cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, 'ps3-1-a-1.png'), D_L)
    cv2.imwrite(os.path.join(output_dir, 'ps3-1-a-2.png'), D_R)
    
    
    # 2
    # Apply disparity_ssd() to pair1-L.png and pair1-R.png (in both directions)
    
    Lorig = np.float32(cv2.imread(os.path.join('input', 'pair1-L.png'), 0)) * (1 / 255.0)  # grayscale, scale to [0.0, 1.0]
    Rorig = np.float32(cv2.imread(os.path.join('input', 'pair1-R.png'), 0)) * (1 / 255.0)
    k = 9
    sigma = 0.3
        
    L = Lorig.copy()
    R = Rorig.copy()
    D_L = disparity_ssd(L, R, k=k)
    D_R = disparity_ssd(R, L, k=k)
    D_L = custom_filter(D_L, sigma=sigma, k=3)
    D_R = custom_filter(D_R, sigma=sigma, k=3)
    cv2.imwrite(os.path.join(output_dir, 'ps3-2-a-1.png'), D_L)
    cv2.imwrite(os.path.join(output_dir, 'ps3-2-a-2.png'), D_R)
        
    # 3
    # Apply disparity_ssd() to noisy versions of pair1 images
    L = Lorig.copy() + np.float32(np.random.normal(0.0, 0.03, Lorig.shape))
    R = Rorig.copy()
    D_L = disparity_ssd(L, R, k=k)
    D_R = disparity_ssd(R, L, k=k)
    D_L = custom_filter(D_L, sigma=sigma, k=3)
    D_R = custom_filter(D_R, sigma=sigma, k=3)
    cv2.imwrite(os.path.join(output_dir, 'ps3-3-a-1.png'), D_L)
    cv2.imwrite(os.path.join(output_dir, 'ps3-3-a-2.png'), D_R)
    
    # contrast
    L = Lorig.copy() * 1.1
    R = Rorig.copy()
    D_L = disparity_ssd(L, R, k=k)
    D_R = disparity_ssd(R, L, k=k)
    D_L = custom_filter(D_L, sigma=sigma, k=3)
    D_R = custom_filter(D_R, sigma=sigma, k=3)
    cv2.imwrite(os.path.join(output_dir, 'ps3-3-b-1.png'), D_L)
    cv2.imwrite(os.path.join(output_dir, 'ps3-3-b-2.png'), D_R)
    
    # 4
    # Implement disparity_ncorr() and apply to pair1 images (original, noisy and contrast-boosted)
    Lorig = np.float32(cv2.imread(os.path.join('input', 'pair1-L.png'), 0)) * (1 / 255.0)  # grayscale, scale to [0.0, 1.0]
    Rorig = np.float32(cv2.imread(os.path.join('input', 'pair1-R.png'), 0)) * (1 / 255.0)
    k = 9
    sigma = 0.3
    
    L = Lorig.copy()
    R = Rorig.copy()
    D_L = disparity_ncorr(L, R, k=k)
    D_R = disparity_ncorr(R, L, k=k)
    D_L = custom_filter(D_L, sigma=sigma, k=3)
    D_R = custom_filter(D_R, sigma=sigma, k=3)
    cv2.imwrite(os.path.join(output_dir, 'ps3-4-a-1.png'), D_L)
    cv2.imwrite(os.path.join(output_dir, 'ps3-4-a-2.png'), D_R)
    
    # noisy
    L = Lorig.copy() + np.float32(np.random.normal(0.0, 0.03, Lorig.shape))
    R = Rorig.copy() 
    D_L = disparity_ncorr(L, R, k=k)
    D_R = disparity_ncorr(R, L, k=k)
    D_L = custom_filter(D_L, sigma=sigma, k=3)
    D_R = custom_filter(D_R, sigma=sigma, k=3)
    cv2.imwrite(os.path.join(output_dir, 'ps3-4-b-1.png'), D_L)
    cv2.imwrite(os.path.join(output_dir, 'ps3-4-b-2.png'), D_R)
    
    # contrast
    L = Lorig.copy() * 1.1
    R = Rorig.copy()
    D_L = disparity_ncorr(L, R, k=k)
    D_R = disparity_ncorr(R, L, k=k)
    D_L = custom_filter(D_L, sigma=sigma, k=3)
    D_R = custom_filter(D_R, sigma=sigma, k=3)
    cv2.imwrite(os.path.join(output_dir, 'ps3-4-b-3.png'), D_L)
    cv2.imwrite(os.path.join(output_dir, 'ps3-4-b-4.png'), D_R)
    
    # 5
    # Apply stereo matching to pair2 images, try pre-processing the images for best results
    Lorig = np.float32(cv2.imread(os.path.join('input', 'pair2-L.png'), 0)) * (1 / 255.0)  # grayscale, scale to [0.0, 1.0]
    Rorig = np.float32(cv2.imread(os.path.join('input', 'pair2-R.png'), 0)) * (1 / 255.0)
    k = 13
    sigma = 0.1
    
    L = Lorig.copy() - 0.2 * cv2.GaussianBlur(Lorig.copy(), (9,9), 0)
    R = Rorig.copy() - 0.2 * cv2.GaussianBlur(Rorig.copy(), (9,9), 0)
    D_L = disparity_ncorr(L, R, k=k)
    D_R = disparity_ncorr(R, L, k=k)
    D_L = custom_filter2(D_L, sigma=sigma)
    D_R = custom_filter2(D_R, sigma=sigma)
    cv2.imwrite(os.path.join(output_dir, 'ps3-5-a-1.png'), D_L)
    cv2.imwrite(os.path.join(output_dir, 'ps3-5-a-2.png'), D_R)

if __name__ == "__main__":
    main()
