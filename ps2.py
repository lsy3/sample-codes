"""Problem Set 2: Edges and Lines."""

import numpy as np
import cv2

import os
import math
from math import pi

input_dir = "input"  # read images from os.path.join(input_dir, <filename>)
output_dir = "output"  # write images to os.path.join(output_dir, <filename>)
HIGHLIGHT_COLOR = (0, 255, 0)
HIGHLIGHT_THICK = 2

def hough_lines_acc(img_edges, rho_res=1, theta_res=pi/90):
    """Compute Hough Transform for lines on edge image.

    Parameters
    ----------
        img_edges: binary edge image
        rho_res: rho resolution (in pixels)
        theta_res: theta resolution (in radians)

    Returns
    -------
        H: Hough accumulator array
        rho: vector of rho values, one for each row of H
        theta: vector of theta values, one for each column of H
    """
    threshold = 100
    img_edges_d = math.sqrt(img_edges.shape[0]**2 + img_edges.shape[1]**2)
    H_rho_size = int(2 * int(img_edges_d) / rho_res) + 1# (d - -d) / rho_res + 1
    H_theta_size = int(pi / theta_res) + 1 # (90 - -90) / (theta_res / pi * 180) + 1
    H = np.zeros((H_rho_size, H_theta_size), dtype=np.uint16)
    rho = np.linspace(-int(img_edges_d), int(img_edges_d), num=H_rho_size)
    theta = np.linspace(-pi/2.0, pi/2.0, num = H_theta_size)
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    H_theta = np.arange(H_theta_size) # np.digitize(theta, theta) - 1
    for y, x in np.transpose(np.where(img_edges > threshold)):
        d_raw = y * sin_theta + x * cos_theta
        H_d = np.digitize(d_raw, rho) - 1
        H[H_d, H_theta] += 1

    return H, rho, theta


def hough_peaks(H, Q, removeNeighbor=0, threshold=[-1,-1]):
    """Find peaks (local maxima) in accumulator array.

    Parameters
    ----------
        H: Hough accumulator array
        Q: number of peaks to find (max)

    Returns
    -------
        peaks: Px2 matrix (P <= Q) where each row is a (rho_idx, theta_idx) pair
    """
    #threshold = H.min() + int((H.max() - H.min()) * 0.50) # 50% of max
    if threshold[0] == -1:
        threshold[0] = H.mean()
    if threshold[1] == -1:
        threshold[1] = H.max() + 1
    i = np.where((H >= threshold[0]) & (H < threshold[1]))
    orderIndex = np.argsort(H[i])[-1:-(Q+1):-1]
    iT = np.transpose(i)
    
    orderIndex2 = []
    for a in orderIndex:
        found = False
        for b in orderIndex2:
            if abs(iT[a][0] - iT[b][0]) <= removeNeighbor and abs(iT[a][1] - iT[b][1]) <= removeNeighbor:
                found = True
                break
        if found:
            continue
        else:
            orderIndex2.append(a)
    peaks = iT[orderIndex2]
    return peaks


def hough_lines_draw(img_out, peaks, rho, theta):
    """Draw lines on an image corresponding to accumulator peaks.

    Parameters
    ----------
        img_out: 3-channel (color) image
        peaks: Px2 matrix where each row is a (rho_idx, theta_idx) index pair
        rho: vector of rho values, such that rho[rho_idx] is a valid rho value
        theta: vector of theta values, such that theta[theta_idx] is a valid theta value
    """
    row, col = img_out.shape[0] - 1, img_out.shape[1] - 1
    
    for rho_i,theta_i in peaks:
        t_rho = rho[rho_i]
        t_theta = theta[theta_i]
        t_costheta = math.cos(t_theta)
        t_sintheta = math.sin(t_theta)
        pts = set()
        # d = x cos(theta) + y sin (theta)
        if math.fabs(t_sintheta) < 1e-5:
            pts.add((int(round(t_rho)), 0))
            pts.add((int(round(t_rho)), row))
        elif math.fabs(t_costheta) < 1e-5:
            pts.add((0, int(round(t_rho))))
            pts.add((col, int(round(t_rho))))
        else:
            # x = 0
            t1 = int(round(t_rho / t_sintheta))
            if t1 >= 0 and t1 <= row: pts.add((0, t1))
            # x = col
            t2 = int(round((t_rho - col * t_costheta) / t_sintheta))
            if t2 >= 0 and t2 <= row: pts.add((col, t2))
            # y = 0
            t3 = int(round(t_rho / t_costheta))
            if t3 >= 0 and t3 <= col: pts.add((t3, 0))
            # y = row
            t4 = int(round((t_rho - row * t_sintheta) / t_costheta))
            if t4 >= 0 and t4 <= col: pts.add((t4, row))
            
        if len(pts) != 2:
            print 'error', pts
            print '(0,{t1}),({col},{t2}),({t3},0),({t4},{row})'.format(col=col,row=row,t1=t1,t2=t2,t3=t3,t4=t4)
        else:
            cv2.line(img_out, pts.pop(), pts.pop(), HIGHLIGHT_COLOR, HIGHLIGHT_THICK)
    pass

def tweeks6c(peaks, rho, theta, dist=50, slope=0.1):
    r = rho[peaks[:,0]]
    t = theta[peaks[:,1]]
    # d = x cos theta + y sin theta
    # y = mx + b format
    m = - np.cos(t) / np.sin(t)
    b = r / np.cos(t)
    
    # pair parallel lines
    i1 = np.argsort(b)
    i2 = set()
    i = 0
    for i in xrange(i1.size-1):
        for j in xrange(i+1,i1.size):
            # measure distance between lines by the x intercept
            if abs(b[i1[i]] - b[i1[j]]) > dist:
                break
            # check if the same slope
            if abs(m[i1[i]] - m[i1[j]]) < slope:
                i2.add(i1[i])
                i2.add(i1[j])      
    return peaks[list(i2),:]

def hough_circles_acc(img_edges, r, sobelX, sobelY):
    """Compute Hough Transform for circles on edge image.

    Parameters
    ----------
        img_edges: binary edge image
        r: radius
        sobelX: gradient on X axis
        sobelY: gradient on Y axis
        rho_res: rho resolution (in pixels)
        theta_res: theta resolution (in radians)

    Returns
    -------
        H: Hough accumulator array
    """
    threshold = 50
    H = np.zeros(img_edges.shape[:2], dtype=np.uint16)
    lenY = img_edges.shape[0]
    lenX = img_edges.shape[1]
    
    for y, x in np.transpose(np.where(img_edges > threshold)):
        theta = math.atan2(sobelY[y,x],sobelX[y,x])
        a = x + r*math.cos(theta)
        b = y + r*math.sin(theta)
        #print "({0},{1}) dY={2:0.1f} dX={3:0.1f} theta={4:0.1f} a={5} b={6}".format(y, x, sobelY[y,x], sobelX[y,x], theta / pi * 180., a, b)
        if b >= 0 and b < lenY and a >= 0 and a < lenX:
            H[b,a] += 1
        a = x - r*math.cos(theta)
        b = y - r*math.sin(theta)
        if b >= 0 and b < lenY and a >= 0 and a < lenX:
            H[b,a] += 1

    return H

def find_circles(img_edges, (rLo, rHi), sobelX, sobelY, Q=10, removeNeighbor=0):
    buff = np.ndarray((0,4), dtype=np.uint16)
    for r in xrange(rLo,rHi+1):
        H = hough_circles_acc(img_edges, r, sobelX, sobelY)
        threshold = H.mean()
        i = np.where(H > threshold)
        oI = np.argsort(H[i])[-1:-(Q+1):-1]
        iT = np.transpose(i)
        H2 = H[i[0][oI],i[1][oI]]
        buff2 = np.concatenate((iT[oI], np.ones((Q,1),dtype=np.uint8)*r, H2.reshape(H2.size,1)), axis=1)
        buff = np.concatenate((buff, buff2), axis=0)
        
    buffOrderIndex = np.argsort(buff[:,3])[-1:-(Q+1):-1]
    buffOrderIndex2 = []
    for a in buffOrderIndex:
        found = False
        for b in buffOrderIndex2:
            if abs(buff[a,0] - buff[b,0]) <= removeNeighbor and abs(buff[a,1] - buff[b,1]) <= removeNeighbor:
                found = True
                break
        if found:
            continue
        else:
            buffOrderIndex2.append(a)
    centers = buff[buffOrderIndex2, 0:2]
    radii = buff[buffOrderIndex2, 2:3]
    
    return centers, radii

def find_circles7b(img_edges, (rLo, rHi), sobelX, sobelY, Q=10, removeNeighbor=0, checkThreshold=180):
    buff = np.ndarray((0,4), dtype=np.uint16)
    for r in xrange(rLo,rHi+1):
        H = hough_circles_acc(img_edges, r, sobelX, sobelY)
        threshold = H.mean()
        i = np.where(H > threshold)
        oI = np.argsort(H[i])[-1:-(Q+1):-1]
        iT = np.transpose(i)
        H2 = H[i[0][oI],i[1][oI]]
        buff2 = np.concatenate((iT[oI], np.ones((Q,1),dtype=np.uint8)*r, H2.reshape(H2.size,1)), axis=1)
        buff = np.concatenate((buff, buff2), axis=0)
        
    buffOrderIndex = np.argsort(buff[:,3])[-1:-(Q+1):-1]
    
    buffOrderIndex2 = []
    for a in buffOrderIndex:
        y,x,r = buff[a,0:3]
        pixSum = 0
        theta = np.arange(-180.0, 180.0) / 180.0 * pi
        for r1 in xrange(r-1,r+2):
            x1 = x - np.int32(r1 * np.cos(theta))
            y1 = y - np.int32(r1 * np.sin(theta))
            t1 = np.where((x1 < img_edges.shape[1]) & (x >= 0))
            x1 = x1[t1]
            y1 = y1[t1]
            t2 = np.where((y1 < img_edges.shape[0]) & (y >= 0))
            x1 = x1[t2]
            y1 = y1[t2]
            pixSum += np.where(img_edges[y1, x1] > 100)[0].size
        if pixSum >= checkThreshold:
            buffOrderIndex2.append(a)
            
    buffOrderIndex3 = []
    for a in buffOrderIndex2:
        found = False
        for b in buffOrderIndex3:
            if abs(buff[a,0] - buff[b,0]) <= removeNeighbor and abs(buff[a,1] - buff[b,1]) <= removeNeighbor:
                found = True
                break
        if found:
            continue
        else:
            buffOrderIndex3.append(a)
    centers = buff[buffOrderIndex3, 0:2]
    radii = buff[buffOrderIndex3, 2:3]
    
    return centers, radii
    
def main():
    """Run code/call functions to solve problems."""

    # 1-a
    # Load the input grayscale image
    img = cv2.imread(os.path.join(input_dir, 'ps2-input0.png'), 0)  # flags=0 ensures grayscale

    img_edges = cv2.Canny(img, 200, 250, apertureSize=3)
    cv2.imwrite(os.path.join(output_dir, 'ps2-1-a-1.png'), img_edges)  # save as ps2-1-a-1.png

    # 2-a
    # Compute Hough Transform for lines on edge image
    H, rho, theta = hough_lines_acc(img_edges, rho_res=1, theta_res=pi/90)

    # Note: Write a normalized uint8 version, mapping min value to 0 and max to 255
    H1_uint8 = np.ndarray(H.shape, dtype=H.dtype)
    cv2.normalize(H,H1_uint8,0,255,cv2.NORM_MINMAX)
    H1_uint8 = np.uint8(H1_uint8)
    cv2.imwrite(os.path.join(output_dir, 'ps2-2-a-1.png'), H1_uint8)

    # 2-b
    # Find peaks (local maxima) in accumulator array
    peaks = hough_peaks(H, 10, removeNeighbor=5)

    H2_uint8 = cv2.cvtColor(H1_uint8, cv2.COLOR_GRAY2BGR)
    for y,x in peaks:
        cv2.circle(H2_uint8, (x,y), 3, HIGHLIGHT_COLOR, thickness=HIGHLIGHT_THICK, lineType=1)
    cv2.imwrite(os.path.join(output_dir, 'ps2-2-b-1.png'), H2_uint8)

    # 2-c
    # Draw lines corresponding to accumulator peaks
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    hough_lines_draw(img_out, peaks, rho, theta)
    cv2.imwrite(os.path.join(output_dir, 'ps2-2-c-1.png'), img_out)  # save as ps2-2-c-1.png

    # 3-a
    # Read ps2-input0-noise.png, compute smoothed image using a Gaussian filter
    img1 = cv2.imread(os.path.join(input_dir, 'ps2-input0-noise.png'), 0)
    img2 = cv2.GaussianBlur(img1, (5,5), 0)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-a-1.png'), img2)    

    # 3-b
    # Compute binary edge images for both original image and smoothed version
    img1_edges = cv2.Canny(img1, 250, 254, apertureSize=3)
    img2_edges = cv2.Canny(img2, 200, 250, apertureSize=3)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-b-1.png'), img1_edges) 
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-b-2.png'), img2_edges) 
    
    # 3-c
    H, rho, theta = hough_lines_acc(img2_edges, rho_res=1, theta_res=pi/90)
    peaks = hough_peaks(H, 20, removeNeighbor=5)
    
    H_uint8 = np.ndarray(H.shape, dtype=H.dtype)
    cv2.normalize(H,H_uint8,0,255,cv2.NORM_MINMAX)
    H_uint8 = np.uint8(H_uint8)
    H_uint8 = cv2.cvtColor(H_uint8, cv2.COLOR_GRAY2BGR)
    for y,x in peaks:
        cv2.circle(H_uint8, (x,y), 3, HIGHLIGHT_COLOR, thickness=HIGHLIGHT_THICK, lineType=1)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-c-1.png'), H_uint8)
    
    img_out = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    hough_lines_draw(img_out, peaks, rho, theta)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-c-2.png'), img_out)

    # 4
    # Like problem 3 above, but using ps2-input1.png
    img1 = cv2.imread(os.path.join(input_dir, 'ps2-input1.png'), 0)
    img2 = cv2.GaussianBlur(img1, (5,5), 0)
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-a-1.png'), img2)    

    img2_edges = cv2.Canny(img2, 100, 200, apertureSize=3)
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-b-1.png'), img2_edges) 
    
    H, rho, theta = hough_lines_acc(img2_edges, rho_res=1, theta_res=pi/90)
    peaks = hough_peaks(H, 10, removeNeighbor=10)
    
    H_uint8 = np.ndarray(H.shape, dtype=H.dtype)
    cv2.normalize(H,H_uint8,0,255,cv2.NORM_MINMAX)
    H_uint8 = np.uint8(H_uint8)
    H_uint8 = cv2.cvtColor(H_uint8, cv2.COLOR_GRAY2BGR)
    for y,x in peaks:
        cv2.circle(H_uint8, (x,y), 3, HIGHLIGHT_COLOR, thickness=HIGHLIGHT_THICK, lineType=1)
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-c-1.png'), H_uint8)
    
    img_out = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    hough_lines_draw(img_out, peaks, rho, theta)
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-c-2.png'), img_out)

    # 5
    # Implement Hough Transform for circles
    img1 = cv2.imread(os.path.join(input_dir, 'ps2-input1.png'), 0)
    img2 = cv2.GaussianBlur(img1, (7,7), 0)
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-1.png'), img2)    

    img2_edges = cv2.Canny(img2, 100, 200, apertureSize=3)
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-2.png'), img2_edges)
    
    img2 = np.float32(img2)
    sobelX = cv2.Sobel(img2,-1,1,0,ksize=1)
    sobelY = cv2.Sobel(img2,-1,0,1,ksize=1)
    
    r = 20
    H = hough_circles_acc(img2_edges, r, sobelX, sobelY)
    centers = hough_peaks(H, 10, removeNeighbor=3)

    img3 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    for y,x in centers:
        cv2.circle(img3, (x,y), r, HIGHLIGHT_COLOR, thickness=HIGHLIGHT_THICK, lineType=1)
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-3.png'), img3)
    
    centers, radii = find_circles(img2_edges, (20, 50), sobelX, sobelY, Q=100, removeNeighbor=5)
    
    img3 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    for y,x,r in np.concatenate((centers,radii),axis=1):
        cv2.circle(img3, (x,y), r, HIGHLIGHT_COLOR, thickness=HIGHLIGHT_THICK, lineType=1)
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-b-1.png'), img3)
    
    # 6
    # Find lines a more realtistic image, ps2-input2.png
    img1 = cv2.imread(os.path.join(input_dir, 'ps2-input2.png'), 0)
    img2 = cv2.GaussianBlur(img1, (5,5), 0)
    img2_edges = cv2.Canny(img2, 100, 200, apertureSize=3)
    
    H, rho, theta = hough_lines_acc(img2_edges, rho_res=1, theta_res=pi/180)
    
    peaks = hough_peaks(H, 15, removeNeighbor=10)
    img_out = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    hough_lines_draw(img_out, peaks, rho, theta)
    cv2.imwrite(os.path.join(output_dir, 'ps2-6-a-1.png'), img_out)
    
    peaks = tweeks6c(peaks, rho, theta, dist=50, slope=1.5)
    img_out = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    hough_lines_draw(img_out, peaks, rho, theta)
    cv2.imwrite(os.path.join(output_dir, 'ps2-6-c-1.png'), img_out)

    # 7
    # Find circles in the same realtistic image, ps2-input2.png
    img1 = cv2.imread(os.path.join(input_dir, 'ps2-input2.png'), 0)
    img2 = cv2.GaussianBlur(img1, (5,5), 0)
    img2_edges = cv2.Canny(img2, 50, 200, apertureSize=3)
    
    img2 = np.float32(img2)
    sobelX = cv2.Sobel(img2,-1,1,0,ksize=1)
    sobelY = cv2.Sobel(img2,-1,0,1,ksize=1)
    centers, radii = find_circles7b(img2_edges, (20, 50), sobelX, sobelY, Q=150, removeNeighbor=10, checkThreshold=180)
    
    img3 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    for y,x,r in np.concatenate((centers,radii),axis=1):
        cv2.circle(img3, (x,y), r, HIGHLIGHT_COLOR, thickness=HIGHLIGHT_THICK, lineType=1)
    cv2.imwrite(os.path.join(output_dir, 'ps2-7-a-1.png'), img3)

    # 8
    # Find lines and circles in distorted image, ps2-input3.png
    img1 = cv2.imread(os.path.join(input_dir, 'ps2-input3.png'), 0)
    img2 = cv2.GaussianBlur(img1, (3,3), 0)
    img2_edges = cv2.Canny(img2, 70, 100, apertureSize=3)
    
    H, rho, theta = hough_lines_acc(img2_edges, rho_res=1, theta_res=pi/180)
    peaks = hough_peaks(H, 10, removeNeighbor=10)
    img_out = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    hough_lines_draw(img_out, peaks, rho, theta)
    
    img2 = np.float32(img2)
    sobelX = cv2.Sobel(img2,-1,1,0,ksize=1)
    sobelY = cv2.Sobel(img2,-1,0,1,ksize=1)
    centers, radii = find_circles(img2_edges, (20, 50), sobelX, sobelY, Q=100, removeNeighbor=20)
    for y,x,r in np.concatenate((centers,radii),axis=1):
        cv2.circle(img_out, (x,y), r, HIGHLIGHT_COLOR, thickness=HIGHLIGHT_THICK, lineType=1)
        
    cv2.imwrite(os.path.join(output_dir, 'ps2-8-a-1.png'), img_out) 

    img_out = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    
    # manual point B to point A transformation
    ylen, xlen = (430, 610)
    img11 = np.zeros((ylen,xlen), dtype=np.uint8)
    for y in xrange(ylen):
        x = np.arange(0, xlen)
        yy = y * 260 / ylen + 31 - x * 10 / xlen
        xx = 110 * (ylen - y) / 430 + x * (400 + 210 * y / ylen) / xlen
        img11[y,:] = img1[yy, xx]
    
    # Find circles
    img2 = cv2.GaussianBlur(img11, (7,7), 0)
    img2_edges = cv2.Canny(img2, 50, 110, apertureSize=3)
    img2 = np.float32(img2)
    sobelX = cv2.Sobel(img2,-1,1,0,ksize=1)
    sobelY = cv2.Sobel(img2,-1,0,1,ksize=1)
    centers, radii = find_circles7b(img2_edges, (20, 50), sobelX, sobelY, Q=150, removeNeighbor=20, checkThreshold=100)
    img_out2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for y,x,r in np.concatenate((centers,radii),axis=1):
        cv2.circle(img_out2, (x,y), r, HIGHLIGHT_COLOR, thickness=HIGHLIGHT_THICK, lineType=1)
    cv2.imwrite(os.path.join(output_dir, 'ps2-8-c-2.png'), img_out2)
    
    # manual point A to point B transformation
    ylen, xlen = (430, 610)
    img11 = np.zeros((ylen,xlen), dtype=np.uint8)
    for y in xrange(ylen):
        x = np.arange(0, xlen)
        yy = y * 260 / ylen + 31 - x * 10 / xlen
        xx = 110 * (ylen - y) / 430 + x * (400 + 210 * y / ylen) / xlen
        img_out[yy, xx] = img_out2[y,:]
    
    
    # Find lines
    img2 = cv2.GaussianBlur(img1, (3,3), 0)
    img2_edges = cv2.Canny(img2, 70, 100, apertureSize=3)
    H, rho, theta = hough_lines_acc(img2_edges, rho_res=1, theta_res=pi/180)
    peaks = hough_peaks(H, 10, removeNeighbor=10)
    hough_lines_draw(img_out, peaks, rho, theta)
        
    cv2.imwrite(os.path.join(output_dir, 'ps2-8-c-1.png'), img_out)

if __name__ == "__main__":
    main()
