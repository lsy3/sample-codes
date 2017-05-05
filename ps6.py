"""Problem Set 6: Optic Flow."""

import numpy as np
import cv2

import os
import __builtin__

# I/O directories
input_dir = "input"
output_dir = "output"

DEBUG = False # debug mode

# Assignment code
def optic_flow_LK(A, B, wsize=11, threshold=1e-10):
    """Compute optic flow using the Lucas-Kanade method.

    Parameters
    ----------
        A: floating-point image, values in [0.0, 1.0]
        B: floating-point image, values in [0.0, 1.0]
        wsize: window size
        threshold: threshold for valid "corners"

    Returns
    -------
        U: raw displacement (in pixels) along X-axis, same size as image, floating-point type
        V: raw displacement (in pixels) along Y-axis, same size and type as U
    """
    
    # initialization for (At*A) * d = At*b
    Ix = cv2.GaussianBlur(A, (3,3), 1)
    Ix = cv2.Sobel(A, -1, 1, 0, ksize=3) / 8.0 # note that divide by 8 only works for ksize = 3
    #Ix[:,1:] -= A[:,:-1]
    #Ix[:,0] *= -1
    Iy = cv2.GaussianBlur(A, (3,3), 1)
    Iy = cv2.Sobel(A, -1, 0, 1, ksize=3) / 8.0 # note that divide by 8 only works for ksize = 3
    #Iy[1:,:] -= A[:-1,:]
    #Iy[0,:] *= -1
    It = B - A

    window = np.ones((wsize,wsize), dtype=np.float_) 
    Ixx = cv2.filter2D(Ix**2, -1, window) 
    Iyy = cv2.filter2D(Iy**2, -1, window)
    Ixy = cv2.filter2D(Ix*Iy, -1, window)
    Ixt = -1.0 * cv2.filter2D(Ix*It, -1, window) #negated
    Iyt = -1.0 * cv2.filter2D(Iy*It, -1, window) #negated
    
    if len(Ix.shape) > 2: # color
        Ixx = Ixx.sum(axis=2)
        Iyy = Iyy.sum(axis=2)
        Ixy = Ixy.sum(axis=2)
        Ixt = Ixt.sum(axis=2)
        Iyt = Iyt.sum(axis=2)
        
    # solve for U, V
    U = np.zeros(Ixx.shape, dtype=np.float_)
    V = np.zeros(Ixx.shape, dtype=np.float_)
    for j in xrange(Ixx.shape[0]):
        for i in xrange(Ixx.shape[1]):
            AtA = np.matrix([[Ixx[j,i],Ixy[j,i]],
                             [Ixy[j,i],Iyy[j,i]]],
                             dtype=np.float_)
            R = np.linalg.det(AtA) - 0.04 * np.trace(AtA)**2
            if R <= threshold:
                U[j,i] = 0
                V[j,i] = 0
            else:
                Atb = np.matrix([[Ixt[j,i]],[Iyt[j,i]]],
                                dtype=np.float_)
                b = np.linalg.inv(AtA) * Atb
                U[j,i] = b[0,0]
                V[j,i] = b[1,0]
                
    return U, V

def reduce(image):
    """Reduce image to the next smaller level.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        reduced_image: same type as image, half size
    """
    kernel = np.matrix([[1,4,6,4,1]]) / 16.0
    kernel = np.outer(kernel, kernel)
    reduced_image = cv2.filter2D(image, -1, kernel)[::2,::2]
    #reduced_image = cv2.GaussianBlur(image, (5,5), 1)[::2,::2]
    return reduced_image


def gaussian_pyramid(image, levels):
    """Create a Gaussian pyramid of given image.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]
        levels: number of levels in the resulting pyramid

    Returns
    -------
        g_pyr: Gaussian pyramid, with g_pyr[0] = image
    """
    g_pyr = [image.copy()]
    for i in xrange(1,levels):
        g_pyr.append(reduce(g_pyr[i-1]))
    return g_pyr


def expand(image):
    """Expand image to the next larger level.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        reduced_image: same type as image, double size
    """
    kernel = np.matrix([[0.125,0.5,0.75,0.5,0.125]])
    kernel = np.outer(kernel, kernel)
    if len(image.shape) == 2:
        nShape = (2*image.shape[0], 2*image.shape[1])
    else: #color
        nShape = (2*image.shape[0], 2*image.shape[1],3)
    expanded_image = np.zeros(nShape, dtype=image.dtype)
    expanded_image[::2,::2] = image
    expanded_image = cv2.filter2D(expanded_image, -1, kernel)
    return expanded_image


def laplacian_pyramid(g_pyr):
    """Create a Laplacian pyramid from a given Gaussian pyramid.

    Parameters
    ----------
        g_pyr: Gaussian pyramid, as returned by gaussian_pyramid()

    Returns
    -------
        l_pyr: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1]
    """
    l_pyr = [i.copy() for i in g_pyr]
    for i in xrange(len(l_pyr)-2,-1,-1):
        eImg = expand(g_pyr[i+1])
        yLen, xLen = l_pyr[i].shape[0], l_pyr[i].shape[1]
        l_pyr[i] -= eImg[:yLen,:xLen]
    return l_pyr


def warp(image, U, V):
    """Warp image using X and Y displacements (U and V).

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        warped: warped image, such that warped[y, x] = image[y + V[y, x], x + U[y, x]]

    """
    mapX, mapY = np.meshgrid(np.arange(image.shape[1], dtype=np.float32),
                             np.arange(image.shape[0], dtype=np.float32))
    mapX = np.float32(mapX + U)
    maxX, maxY = image.shape[1] - 1, image.shape[0] - 1
    mapX[np.where(mapX>maxX)] = maxX
    mapX[np.where(mapX<0)] = 0
    mapY = np.float32(mapY + V)
    mapY[np.where(mapY>maxY)] = maxY
    mapY[np.where(mapY<0)] = 0
    warped = cv2.remap(image, mapX, mapY, cv2.INTER_LINEAR)
    return warped


def hierarchical_LK(A, B, n=4, wsize=11, threshold=1e-10):
    """Compute optic flow using the Hierarchical Lucas-Kanade method.

    Parameters
    ----------
        A: grayscale floating-point image, values in [0.0, 1.0]
        B: grayscale floating-point image, values in [0.0, 1.0]
        n: max level of gaussian pyramid
        wsize: window size
        threshold: threshold for valid "corners"
        
    Returns
    -------
        U: raw displacement (in pixels) along X-axis, same size as image, floating-point type
        V: raw displacement (in pixels) along Y-axis, same size and type as U
    """
    A_g_pyr = gaussian_pyramid(A, n)
    B_g_pyr = gaussian_pyramid(B, n)
    # cv2.imwrite(os.path.join(output_dir, 'A_g_pyr.png'), pyramid_image(A_g_pyr))
    # cv2.imwrite(os.path.join(output_dir, 'B_g_pyr.png'), pyramid_image(B_g_pyr))
    
    for k in xrange(n-1,-1,-1):
        Ak, Bk = A_g_pyr[k], B_g_pyr[k]
        if k == (n-1):
            U = np.zeros(Ak.shape[:2], dtype=np.float_)
            V = np.zeros(Ak.shape[:2], dtype=np.float_)
        else:
            U = (2.0*expand(U))[0:Bk.shape[0],0:Bk.shape[1]]
            V = (2.0*expand(V))[0:Bk.shape[0],0:Bk.shape[1]]
            #U = cv2.resize(2.0 * expand(U), (Bk.shape[1], Bk.shape[0]))
            #V = cv2.resize(2.0 * expand(V), (Bk.shape[1], Bk.shape[0]))
        Ck = warp(Bk, U, V)
        Dx, Dy = optic_flow_LK(Ak, Ck, wsize=wsize, threshold=threshold)
        U += Dx
        V += Dy

    return U, V

def quiver_plot(U, V, stride=10, scale=5, color = (0, 255, 0)):
    """Generate quiver plot

    Parameters
    ----------
        U: raw displacement (in pixels) along X-axis, same size as image, floating-point type
        V: raw displacement (in pixels) along Y-axis, same size and type as U
        stride: plot every so many rows, columns
        scale: scale up vector lengths by this factor
        color: color of arrows
    Returns
    -------
        img_out: quiver plot image, values in [0, 255]
    """
    img_out = np.zeros((V.shape[0], V.shape[1], 3), dtype=np.uint8)
    
    for y in xrange(0, V.shape[0], stride):
        for x in xrange(0, V.shape[1], stride):
            x2 = x + int(U[y,x]*scale)
            y2 = y + int(V[y,x]*scale)
            #cv2.arrowedLine(img_out, (x, y), (x2, y2), color, 
            #                line_type=cv2.CV_AA)
            cv2.arrowedLine(img_out, (x, y), (x2, y2), color)
    
    return img_out
    
def pseudocolor(U, V):
    """Generate pseudocolor or false color image

    Parameters
    ----------
        U: raw displacement (in pixels) along X-axis, same size as image, floating-point type
        V: raw displacement (in pixels) along Y-axis, same size and type as U
    Returns
    -------
        img_out: pseudocolor or false color image, values in [0, 255]
    """
    U2 = np.zeros(U.shape, dtype=U.dtype)
    V2 = np.zeros(V.shape, dtype=V.dtype)
    cv2.normalize(U,U2,0,255,cv2.NORM_MINMAX)
    cv2.normalize(V,V2,0,255,cv2.NORM_MINMAX)
    img_out = np.concatenate((U2, V2), axis=1)
    img_out = cv2.applyColorMap(np.uint8(img_out), cv2.COLORMAP_JET)
    return img_out    

def pyramid_image(g_pyr):
    """Create pyramid image

    Parameters
    ----------
        g_pyr: Gaussian pyramid, values in [0.0, 1.0]

    Returns
    -------
        img_out: python image, values in [0, 255]
    """
    xLen = __builtin__.reduce(lambda x, y: x+y, [i.shape[1] for i in g_pyr])
    yLen = g_pyr[0].shape[0]
    if len(g_pyr[0].shape) == 2:
        img_out = np.zeros((yLen, xLen), dtype=np.uint8)
    else: # color
        img_out = np.zeros((yLen, xLen, 3), dtype=np.uint8)
    
    xPos = 0
    for img in g_pyr:
        img2 = np.zeros(img.shape, dtype=img.dtype)
        cv2.normalize(img,img2,0,255,cv2.NORM_MINMAX)
        img2 = np.uint8(img2)
        img_out[0:img.shape[0],xPos:xPos+img.shape[1]] = img2
        xPos += img.shape[1]
    
    return img_out

def scale(img):
    """Scale values such that zero difference maps to neutral gray, max -ve to black and max +ve to white

    Parameters
    ----------
        img: input image, values in [-1.0, 1.0]

    Returns
    -------
        img_out: output image, values in [0, 255]
    """
    img_out = img.copy()
    print "max={0:0.2f}, min={1:0.2f}, sse={2:0.2f}".format(img_out.max(), img_out.min(), (img_out**2).sum())
    img_out[np.where(img_out>0)] /= img_out.max()
    img_out[np.where(img_out<0)] /= abs(img_out.min())
    cv2.normalize(img_out,img_out,0,255,cv2.NORM_MINMAX)
    img_out = np.uint8(img_out)
    return img_out
   
# Driver code
def main():
    # 1a and 1b
    ShiftName = ['0', 'R2', 'R5U5', 'R10', 'R20', 'R40']
    Shift = {}
    for name in ShiftName:
        Shift[name] = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift{0}.png'.format(name)), 0) / 255.0
        # Optionally, smooth the images if LK doesn't work well on raw images
        Shift[name] = cv2.GaussianBlur(Shift[name], (5,5), 1)

    ShiftPair = [('0','R2',11,'ps6-1-a-1'), #1a
                 ('0','R5U5',41,'ps6-1-a-2'),
                 ('0','R10',41,'ps6-1-b-1'), #1b
                 ('0','R20',41,'ps6-1-b-2'),
                 ('0','R40',41,'ps6-1-b-3')]
                                  
    for img1, img2, wsize, out in ShiftPair:
        U, V = optic_flow_LK(Shift[img1], Shift[img2], wsize=wsize)
        # Save U, V as side-by-side false-color image 
        # or single quiver plot
        cv2.imwrite(os.path.join(output_dir, out+'.png'), 
            pseudocolor(U, V))
        if DEBUG:
            cv2.imwrite(os.path.join(output_dir, out+'b.png'), 
                        quiver_plot(U, V))
    
    # 2a
    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.0
    yos_img_01_g_pyr = gaussian_pyramid(yos_img_01, 4)
    # Save pyramid images as a single side-by-side image
    cv2.imwrite(os.path.join(output_dir, 'ps6-2-a-1.png'), 
                pyramid_image(yos_img_01_g_pyr))

    # 2b
    yos_img_01_l_pyr = laplacian_pyramid(yos_img_01_g_pyr)
    # Save pyramid images as a single side-by-side image
    cv2.imwrite(os.path.join(output_dir, 'ps6-2-b-1.png'), 
                pyramid_image(yos_img_01_l_pyr))
                
    # 3a
    yos_img = []
    yos_img_g_pyr = []
    for i in xrange(0,3):
        img = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_{0:02}.jpg'.format(i+1)), 0) / 255.0
        img = cv2.GaussianBlur(img, (5,5), 1)
        yos_img.append(img)
        yos_img_g_pyr.append(gaussian_pyramid(yos_img[i], 4))
    
    level = 1 # Select appropriate pyramid *level* that leads to best optic flow estimation
    yLen, xLen = yos_img[0].shape[0], yos_img[0].shape[1]
    outA = np.ndarray(shape=(0,2*xLen,3), dtype=np.uint8) # pseudocolor image buffer
    outDiff = np.ndarray(shape=(0,xLen), dtype=np.uint8) # diff between warped and original image buffer
    if DEBUG:
        outB = np.ndarray(shape=(0,xLen,3), dtype=np.uint8) # quiver_plot image buffer
        outC = np.ndarray(shape=(0,xLen), dtype=np.uint8) # source original image buffer
        outD = np.ndarray(shape=(0,xLen), dtype=np.uint8) # warped image buffer
        outE = np.ndarray(shape=(0,xLen), dtype=np.uint8) # destination original image buffer   
    for i in xrange(1,3):
        print "ds1-{0}-{1} level={2}".format(i-1, i, level)
        U, V = optic_flow_LK(yos_img_g_pyr[i-1][level], yos_img_g_pyr[i][level], wsize=21)
        # Scale up U, V to original image size (note: don't forget to scale values as well!)
        U = (2.0**level) * cv2.resize(U, (xLen, yLen))
        V = (2.0**level) * cv2.resize(V, (xLen, yLen))
        # Save U, V as side-by-side false-color image or single quiver plot
        outA = np.concatenate((outA, pseudocolor(U, V)), axis=0)
        warped = warp(yos_img[i], U, V)
        # Save difference image between warped and original
        outDiff = np.concatenate((outDiff, scale(yos_img[i-1] - warped)), axis=0)    
        if DEBUG:
            outB = np.concatenate((outB, quiver_plot(U, V)), axis=0)    
            outC = np.concatenate((outC, np.uint8(yos_img[i] * 255.0)), axis=0)
            outD = np.concatenate((outD, np.uint8(warped * 255.0)), axis=0)
            outE = np.concatenate((outE, np.uint8(yos_img[i-1] * 255.0)), axis=0)

    cv2.imwrite(os.path.join(output_dir, 'ps6-3-a-1.png'), outA)
    cv2.imwrite(os.path.join(output_dir, 'ps6-3-a-2.png'), outDiff)
    if DEBUG:
        cv2.imwrite(os.path.join(output_dir, 'ps6-3-a-1b.png'), outB)
        cv2.imwrite(os.path.join(output_dir, 'ps6-3-a-1c.png'), outC)
        cv2.imwrite(os.path.join(output_dir, 'ps6-3-a-1d.png'), outD)
        cv2.imwrite(os.path.join(output_dir, 'ps6-3-a-1e.png'), outE)

    # Repeat for DataSeq2 (save images)
    ds2_img = []
    ds2_img_g_pyr = []
    for i in xrange(0,3):
        img = cv2.imread(os.path.join(input_dir, 'DataSeq2', '{0}.png'.format(i)), 0) / 255.0
        img = cv2.GaussianBlur(img, (5,5), 1)
        ds2_img.append(img)
        ds2_img_g_pyr.append(gaussian_pyramid(ds2_img[i], 4))
    
    level = 2 # Select appropriate pyramid *level* that leads to best optic flow estimation
    yLen, xLen = ds2_img[0].shape[0], ds2_img[0].shape[1]
    outA = np.ndarray(shape=(0,2*xLen,3), dtype=np.uint8) # pseudocolor image buffer
    outDiff = np.ndarray(shape=(0,xLen), dtype=np.uint8) # diff between warped and original image buffer
    if DEBUG:
        outB = np.ndarray(shape=(0,xLen,3), dtype=np.uint8) # quiver_plot image buffer
        outC = np.ndarray(shape=(0,xLen), dtype=np.uint8) # source original image buffer
        outD = np.ndarray(shape=(0,xLen), dtype=np.uint8) # warped image buffer
        outE = np.ndarray(shape=(0,xLen), dtype=np.uint8) # destination original image buffer
    for i in xrange(1,3):
        print "ds2-{0}-{1} level={2}".format(i-1, i, level)
        U, V = optic_flow_LK(ds2_img_g_pyr[i-1][level], ds2_img_g_pyr[i][level], wsize=11)
        # Scale up U, V to original image size (note: don't forget to scale values as well!)
        U = (2.0**level) * cv2.resize(U, (xLen, yLen))
        V = (2.0**level) * cv2.resize(V, (xLen, yLen))
        # Save U, V as side-by-side false-color image or single quiver plot
        outA = np.concatenate((outA, pseudocolor(U, V)), axis=0)
        warped = warp(ds2_img[i], U, V)
        # Save difference image between warped and original
        outDiff = np.concatenate((outDiff, scale(ds2_img[i-1] - warped)), axis=0)
        if DEBUG:
            outB = np.concatenate((outB, quiver_plot(U, V)), axis=0)
            outC = np.concatenate((outC, np.uint8(ds2_img[i] * 255.0)), axis=0)
            outD = np.concatenate((outD, np.uint8(warped * 255.0)), axis=0)
            outE = np.concatenate((outE, np.uint8(ds2_img[i-1] * 255.0)), axis=0)
    
    cv2.imwrite(os.path.join(output_dir, 'ps6-3-a-3.png'), outA)
    cv2.imwrite(os.path.join(output_dir, 'ps6-3-a-4.png'), outDiff)
    if DEBUG:
        cv2.imwrite(os.path.join(output_dir, 'ps6-3-a-3b.png'), outB)
        cv2.imwrite(os.path.join(output_dir, 'ps6-3-a-3c.png'), outC)
        cv2.imwrite(os.path.join(output_dir, 'ps6-3-a-3d.png'), outD)
        cv2.imwrite(os.path.join(output_dir, 'ps6-3-a-3e.png'), outE)

    # 4a
    ShiftName = ['0', 'R10', 'R20', 'R40']
    Shift = {}
    for name in ShiftName:
        img = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift{0}.png'.format(name)), 0) / 255.0
        # Optionally, smooth the images if LK doesn't work well on raw images
        Shift[name] = cv2.GaussianBlur(img, (5,5), 1)
    
    ShiftSet = [('0', 'R10', int(np.ceil(np.log(10)/np.log(2)))),
                ('0', 'R20', int(np.ceil(np.log(20)/np.log(2)))),
                ('0', 'R40', int(np.ceil(np.log(40)/np.log(2))))]
    yLen, xLen = Shift['0'].shape[0], Shift['0'].shape[1]
    outA = np.ndarray(shape=(0,2*xLen,3), dtype=np.uint8) # pseudocolor image buffer
    outDiff = np.ndarray(shape=(0,xLen), dtype=np.uint8) # diff between warped and original image buffer
    if DEBUG:
        outB = np.ndarray(shape=(0,xLen,3), dtype=np.uint8) # quiver_plot image buffer
        outC = np.ndarray(shape=(0,xLen), dtype=np.uint8) # source original image buffer
        outD = np.ndarray(shape=(0,xLen), dtype=np.uint8) # warped image buffer
        outE = np.ndarray(shape=(0,xLen), dtype=np.uint8) # destination original image buffer
    for imgA, imgB, level in ShiftSet:
        print "Shift{0}-{1} level={2}".format(imgA, imgB, level)
        U, V = hierarchical_LK(Shift[imgA], Shift[imgB], n=level, wsize=7)
        warped = warp(Shift[imgB], U, V)
        print "U=({0:0.2f},{1:0.2f}) V=({2:0.2f},{3:0.2f})".format(U.min(), U.max(), V.min(), V.max())
        # Save displacement image pairs (U, V), stacked
        # Hint: You can use np.concatenate()
        outA = np.concatenate((outA, pseudocolor(U, V)), axis=0)
        # Save difference between each warped image and original image (Shift0), stacked
        outDiff = np.concatenate((outDiff, scale(Shift[imgA] - warped)), axis=0)
        if DEBUG:
            outB = np.concatenate((outB, quiver_plot(U, V, scale=1)), axis=0)
            outC = np.concatenate((outC, np.uint8(Shift[imgB] * 255.0)), axis=0)
            outD = np.concatenate((outD, np.uint8(warped * 255.0)), axis=0)
            outE = np.concatenate((outE, np.uint8(Shift[imgA] * 255.0)), axis=0)
    cv2.imwrite(os.path.join(output_dir, 'ps6-4-a-1.png'), outA)
    cv2.imwrite(os.path.join(output_dir, 'ps6-4-a-2.png'), outDiff)
    if DEBUG:
        cv2.imwrite(os.path.join(output_dir, 'ps6-4-a-1b.png'), outB)
        cv2.imwrite(os.path.join(output_dir, 'ps6-4-a-1c.png'), outC)
        cv2.imwrite(os.path.join(output_dir, 'ps6-4-a-1d.png'), outD)
        cv2.imwrite(os.path.join(output_dir, 'ps6-4-a-1e.png'), outE)
    
    # 4b
    # Repeat for DataSeq1 (use yos_img_01.png as the original)
    ds1Name = ['01', '02', '03']
    ds1 = {}
    for name in ds1Name:
        img = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_{0}.jpg'.format(name)), 0) / 255.0
        # Optionally, smooth the images if LK doesn't work well on raw images
        ds1[name] = cv2.GaussianBlur(img, (5,5), 1)
    
    ds1Set = [('01', '02', int(np.ceil(np.log(10)/np.log(2)))),
              ('02', '03', int(np.ceil(np.log(10)/np.log(2))))]
    yLen, xLen = ds1['01'].shape[0], ds1['01'].shape[1]
    outA = np.ndarray(shape=(0,2*xLen,3), dtype=np.uint8) # pseudocolor image buffer
    outDiff = np.ndarray(shape=(0,xLen), dtype=np.uint8) # diff between warped and original image buffer
    if DEBUG:
        outB = np.ndarray(shape=(0,xLen,3), dtype=np.uint8) # quiver_plot image buffer
        outC = np.ndarray(shape=(0,xLen), dtype=np.uint8) # source original image buffer
        outD = np.ndarray(shape=(0,xLen), dtype=np.uint8) # warped image buffer
        outE = np.ndarray(shape=(0,xLen), dtype=np.uint8) # destination original image buffer
    for imgA, imgB, level in ds1Set:
        print "ds1-{0}-{1} level={2}".format(imgA, imgB, level)
        U, V = hierarchical_LK(ds1[imgA], ds1[imgB], n=level, wsize=21)
        warped = warp(ds1[imgB], U, V)
        print "U=({0:0.2f},{1:0.2f}) V=({2:0.2f},{3:0.2f})".format(U.min(), U.max(), V.min(), V.max())
        # Save displacement image pairs (U, V), stacked
        # Hint: You can use np.concatenate()
        outA = np.concatenate((outA, pseudocolor(U, V)), axis=0)
        # Save difference between each warped image and original image (ds10), stacked
        outDiff = np.concatenate((outDiff, scale(ds1[imgA] - warped)), axis=0)
        if DEBUG:
            outB = np.concatenate((outB, quiver_plot(U, V, scale=3)), axis=0)
            outC = np.concatenate((outC, np.uint8(ds1[imgB] * 255.0)), axis=0)
            outD = np.concatenate((outD, np.uint8(warped * 255.0)), axis=0)
            outE = np.concatenate((outE, np.uint8(ds1[imgA] * 255.0)), axis=0)
    cv2.imwrite(os.path.join(output_dir, 'ps6-4-b-1.png'), outA)
    cv2.imwrite(os.path.join(output_dir, 'ps6-4-b-2.png'), outDiff)
    if DEBUG:
        cv2.imwrite(os.path.join(output_dir, 'ps6-4-b-1b.png'), outB)
        cv2.imwrite(os.path.join(output_dir, 'ps6-4-b-1c.png'), outC)
        cv2.imwrite(os.path.join(output_dir, 'ps6-4-b-1d.png'), outD)
        cv2.imwrite(os.path.join(output_dir, 'ps6-4-b-1e.png'), outE)

    # 4c
    # Repeat for DataSeq2 (use 0.png as the original)
    ds2Name = ['0', '1', '2']
    ds2 = {}
    for name in ds2Name:
        img = cv2.imread(os.path.join(input_dir, 'DataSeq2', '{0}.png'.format(name)), 0) / 255.0
        # Optionally, smooth the images if LK doesn't work well on raw images
        ds2[name] = cv2.GaussianBlur(img, (5,5), 1)
    
    ds2Set = [('0', '1', int(np.ceil(np.log(10)/np.log(2)))),
              ('1', '2', int(np.ceil(np.log(10)/np.log(2))))]
    yLen, xLen = ds2['0'].shape[0], ds2['0'].shape[1]
    outA = np.ndarray(shape=(0,2*xLen,3), dtype=np.uint8) # pseudocolor image buffer
    outDiff = np.ndarray(shape=(0,xLen), dtype=np.uint8) # diff between warped and original image buffer
    if DEBUG:
        outB = np.ndarray(shape=(0,xLen,3), dtype=np.uint8) # quiver_plot image buffer
        outC = np.ndarray(shape=(0,xLen), dtype=np.uint8) # source original image buffer
        outD = np.ndarray(shape=(0,xLen), dtype=np.uint8) # warped image buffer
        outE = np.ndarray(shape=(0,xLen), dtype=np.uint8) # destination original image buffer
    for imgA, imgB, level in ds2Set:
        print "ds2-{0}-{1} level={2}".format(imgA, imgB, level)
        U, V = hierarchical_LK(ds2[imgA], ds2[imgB], n=level, wsize=11)
        warped = warp(ds2[imgB], U, V)
        print "U=({0:0.2f},{1:0.2f}) V=({2:0.2f},{3:0.2f})".format(U.min(), U.max(), V.min(), V.max())
        # Save displacement image pairs (U, V), stacked
        # Hint: You can use np.concatenate()
        outA = np.concatenate((outA, pseudocolor(U, V)), axis=0)
        # Save difference between each warped image and original image (ds20), stacked
        outDiff = np.concatenate((outDiff, scale(ds2[imgA] - warped)), axis=0)
        if DEBUG:
            outB = np.concatenate((outB, quiver_plot(U, V, scale=3)), axis=0)
            outC = np.concatenate((outC, np.uint8(ds2[imgB] * 255.0)), axis=0)
            outD = np.concatenate((outD, np.uint8(warped * 255.0)), axis=0)
            outE = np.concatenate((outE, np.uint8(ds2[imgA] * 255.0)), axis=0)
    cv2.imwrite(os.path.join(output_dir, 'ps6-4-c-1.png'), outA)
    cv2.imwrite(os.path.join(output_dir, 'ps6-4-c-2.png'), outDiff)
    if DEBUG:
        cv2.imwrite(os.path.join(output_dir, 'ps6-4-c-1b.png'), outB)
        cv2.imwrite(os.path.join(output_dir, 'ps6-4-c-1c.png'), outC)
        cv2.imwrite(os.path.join(output_dir, 'ps6-4-c-1d.png'), outD)
        cv2.imwrite(os.path.join(output_dir, 'ps6-4-c-1e.png'), outE)
    
    # 5a
    # Repeat for Juggle Sequence (use 0.png as the original)
    JuggleName = ['0', '1', '2']
    Juggle = {}
    for name in JuggleName:
        img = cv2.imread(os.path.join(input_dir, 'Juggle', '{0}.png'.format(name)), 1) / 255.0
        # cheat: since ball is color yellow. G+R-B to highlight yellow.
        img = img[:,:,1] + img[:,:,2] - img[:,:,0]
        cv2.normalize(img,img,0.0,1.0,cv2.NORM_MINMAX)
        # Optionally, smooth the images if LK doesn't work well on raw images
        Juggle[name] = cv2.GaussianBlur(img, (7,7), 1)
    
    JuggleSet = [('0', '1', int(np.ceil(np.log(20)/np.log(2))), 5), #img1, img2, level, wsize
                 ('1', '2', int(np.ceil(np.log(20)/np.log(2))), 5)]
    yLen, xLen = Juggle['0'].shape[0], Juggle['0'].shape[1]
    outA = np.ndarray(shape=(0,2*xLen,3), dtype=np.uint8) # pseudocolor image buffer
    outDiff = np.ndarray(shape=(0,xLen), dtype=np.uint8) # diff between warped and original image buffer
    if DEBUG:
        outB = np.ndarray(shape=(0,xLen,3), dtype=np.uint8) # quiver_plot image buffer
        tShape = (0,xLen) if len(Juggle['0'].shape) == 2 else (0,xLen,3)
        outC = np.ndarray(shape=tShape, dtype=np.uint8) # source original image buffer
        outD = np.ndarray(shape=tShape, dtype=np.uint8) # warped image buffer
        outE = np.ndarray(shape=tShape, dtype=np.uint8) # destination original image buffer
    for imgA, imgB, level, wsize in JuggleSet:
        print "Juggle-{0}-{1} level={2}, wsize={3}".format(imgA, imgB, level, wsize)
        U, V = hierarchical_LK(Juggle[imgA], Juggle[imgB], n=level, wsize=wsize, threshold=1e-4)
        warped = warp(Juggle[imgB], U, V)
        print "U=({0:0.2f},{1:0.2f}) V=({2:0.2f},{3:0.2f})".format(U.min(), U.max(), V.min(), V.max())
        # Save displacement image pairs (U, V), stacked
        outA = np.concatenate((outA, pseudocolor(U, V)), axis=0)
        # Save difference between each warped image and original image (Juggle0), stacked
        if len(warped.shape) == 2:
            outDiff = np.concatenate((outDiff, scale(Juggle[imgA] - warped)), axis=0)
        else: # color
            outDiff = np.concatenate((outDiff, cv2.cvtColor(scale(Juggle[imgA] - warped),cv2.COLOR_BGR2GRAY)), axis=0)
        if DEBUG:
            outB = np.concatenate((outB, quiver_plot(U, V, scale=3)), axis=0)
            outC = np.concatenate((outC, np.uint8(Juggle[imgB] * 255.0)), axis=0)
            outD = np.concatenate((outD, np.uint8(warped * 255.0)), axis=0)
            outE = np.concatenate((outE, np.uint8(Juggle[imgA] * 255.0)), axis=0)
    cv2.imwrite(os.path.join(output_dir, 'ps6-5-a-1.png'), outA)
    cv2.imwrite(os.path.join(output_dir, 'ps6-5-a-2.png'), outDiff)
    if DEBUG:
        cv2.imwrite(os.path.join(output_dir, 'ps6-5-a-1b.png'), outB)
        cv2.imwrite(os.path.join(output_dir, 'ps6-5-a-1c.png'), outC)
        cv2.imwrite(os.path.join(output_dir, 'ps6-5-a-1d.png'), outD)
        cv2.imwrite(os.path.join(output_dir, 'ps6-5-a-1e.png'), outE)
    
if __name__ == "__main__":
    main()
