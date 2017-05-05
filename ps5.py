"""Problem Set 5: Harris, SIFT, RANSAC."""

import numpy as np
import cv2

import os
import sys

# I/O directories
input_dir = "input"
output_dir = "output"
HIGHLIGHT_COLOR = (0,255,0)
HIGHLIGHT_CYCLE = [(255,0,0), (255,128,0), (255,255,0), (128,255,0), (0,255,0), (0,255,128), (0,255,255), (0,128,255), (0,0,255), (128,0,255), (255,0,255)]
HIGHLIGHT_THICK = 1

# Assignment code
def gradientX(image):
    """Compute image gradient in X direction.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        Ix: image gradient in X direction, values in [-1.0, 1.0]
    """
    Ix = cv2.GaussianBlur(image, (3,3), 0)
    Ix = cv2.Sobel(Ix, -1, 1, 0, ksize=3)
    Ix[np.where(Ix>0)] /= Ix.max()
    Ix[np.where(Ix<0)] /= abs(Ix.min())
    return Ix


def gradientY(image):
    """Compute image gradient in Y direction.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        Iy: image gradient in Y direction, values in [-1.0, 1.0]
    """
    Iy = cv2.GaussianBlur(image, (3,3), 1)
    Iy = cv2.Sobel(Iy, -1, 0, 1, ksize=3)
    Iy[np.where(Iy>0)] /= Iy.max()
    Iy[np.where(Iy<0)] /= abs(Iy.min())
    return Iy


def make_image_pair(image1, image2):
    """Adjoin two images side-by-side to make a single new image.

    Parameters
    ----------
        image1: first image, could be grayscale or color (BGR)
        image2: second image, same type as first

    Returns
    -------
        image_pair: combination of both images, side-by-side, same type
    """
    image_pair = np.concatenate((image1,image2), axis=1)
    return image_pair


def harris_response(Ix, Iy, kernel, alpha):
    """Compute Harris reponse map using given image gradients.

    Parameters
    ----------
        Ix: image gradient in X direction, values in [-1.0, 1.0]
        Iy: image gradient in Y direction, same size and type as Ix
        kernel: 2D windowing kernel with weights, typically square
        alpha: Harris detector parameter multiplied with square of trace

    Returns
    -------
        R: Harris response map, same size as inputs, floating-point
    """
    Sx2 = cv2.filter2D(Ix**2, -1, kernel)
    Sy2 = cv2.filter2D(Iy**2, -1, kernel)
    Sxy = cv2.filter2D(Ix*Iy, -1, kernel)
    R = np.zeros(Sx2.shape, dtype=Sx2.dtype)
    for y in xrange(Sx2.shape[0]):
        for x in xrange(Sx2.shape[1]):
            M = np.array([[Sx2[y,x],Sxy[y,x]],[Sxy[y,x],Sy2[y,x]]])
            R[y,x] = np.linalg.det(M) - alpha * np.trace(M)**2
    # Note: Define any other parameters you need locally or as keyword arguments
    return R


def find_corners(R, threshold, radius):
    """Find corners in given response map.

    Parameters
    ----------
        R: floating-point response map, e.g. output from the Harris detector
        threshold: response values less than this should not be considered plausible corners
        radius: radius of circular region for non-maximal suppression (could be half the side of square instead)

    Returns
    -------
        corners: peaks found in response map R, as a sequence (list) of (x, y) coordinates
    """
    rad = int(round(2*radius - 1) / 2)
    corners = []
    for y in xrange(R.shape[0]):
        for x in xrange(R.shape[1]):
            if R[y,x] <= threshold:
                continue
            yLow = 0 if rad > y else y - rad
            xLow = 0 if rad > x else x - rad
            if R[yLow:y+rad+1,xLow:x+rad+1].max() == R[y,x]:
                corners.append((x,y))
    return corners


def draw_corners(image, corners):
    """Draw corners on (a copy of) given image.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]
        corners: peaks found in response map R, as a sequence (list) of (x, y) coordinates

    Returns
    -------
        image_out: copy of image with corners drawn on it, color (BGR), uint8, values in [0, 255]
    """
    image_out = cv2.cvtColor(np.float32(image), cv2.COLOR_GRAY2BGR)
    cv2.normalize(image_out,image_out,0,255,cv2.NORM_MINMAX)
    image_out = np.uint8(image_out)
    for x, y in corners:
        #print (x, y)
        cv2.circle(image_out,(x,y), 3, HIGHLIGHT_COLOR, thickness=HIGHLIGHT_THICK, lineType=1)
    return image_out


def gradient_angle(Ix, Iy):
    """Compute angle (orientation) image given X and Y gradients.

    Parameters
    ----------
        Ix: image gradient in X direction
        Iy: image gradient in Y direction, same size and type as Ix

    Returns
    -------
        angle: gradient angle image, each value in degrees [0, 359)
    """
    # Note: +ve X axis points to the right (0 degrees), +ve Y axis points down (90 degrees)
    angle = np.mod(np.arctan2(Iy, Ix) * 180 / np.pi + 360.0, 360.0)
    return angle


def get_keypoints(points, R, angle, _size, _octave=0):
    """Create OpenCV KeyPoint objects given interest points, response and angle images.

    Parameters
    ----------
        points: interest points (e.g. corners), as a sequence (list) of (x, y) coordinates
        R: floating-point response map, e.g. output from the Harris detector
        angle: gradient angle (orientation) image, each value in degrees [0, 359)
        _size: fixed _size parameter to pass to cv2.KeyPoint() for all points
        _octave: fixed _octave parameter to pass to cv2.KeyPoint() for all points

    Returns
    -------
        keypoints: a sequence (list) of cv2.KeyPoint objects
    """
    keypoints = []
    for x, y in points:
        keypoints.append(cv2.KeyPoint(x=x, y=y, _size=_size, _angle=angle[y,x], _response=R[y,x], _octave=_octave))
    return keypoints

def draw_keypoints(image, keypoints):
    """Draw keypoints on (a copy of) given image.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]
        keypoints: a sequence (list) of cv2.KeyPoint objects
    Returns
    -------
        image_out: copy of image with corners drawn on it, color (BGR), uint8, values in [0, 255]
    """
    # Note: You should be able to plot the keypoints using cv2.drawKeypoints() in OpenCV 2.4.9+
    image_out = cv2.cvtColor(np.float32(image), cv2.COLOR_GRAY2BGR)
    cv2.normalize(image_out,image_out,0,255,cv2.NORM_MINMAX)
    image_out = np.uint8(image_out)
    image_out = cv2.drawKeypoints(image_out, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return image_out
    
def get_descriptors(image, keypoints):
    """Extract feature descriptors from image at each keypoint.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]
        keypoints: a sequence (list) of cv2.KeyPoint objects

    Returns
    -------
        descriptors: 2D NumPy array of shape (len(keypoints), 128)
    """
    # Note: You can use OpenCV's SIFT.compute() method to extract descriptors, or write your own!
    image2 = image.copy()
    cv2.normalize(image2,image2,0,255,cv2.NORM_MINMAX)
    image2 = np.uint8(image2)
    
    sift = cv2.SIFT()
    keypoints, descriptors = sift.compute(image2, keypoints)
    return descriptors


def match_descriptors(desc1, desc2):
    """Match feature descriptors obtained from two images.

    Parameters
    ----------
        desc1: descriptors from image 1, as returned by SIFT.compute()
        desc2: descriptors from image 2, same format as desc1

    Returns
    -------
        matches: a sequence (list) of cv2.DMatch objects containing corresponding descriptor indices
    """
    # Note: You can use OpenCV's descriptor matchers, or roll your own!
    bfm = cv2.BFMatcher()
    matches = bfm.match(desc1, desc2)
    return matches


def draw_matches(image1, image2, kp1, kp2, matches):
    """Show matches by drawing lines connecting corresponding keypoints.

    Parameters
    ----------
        image1: first image
        image2: second image, same type as first
        kp1: list of keypoints (cv2.KeyPoint objects) found in image1
        kp2: list of keypoints (cv2.KeyPoint objects) found in image2
        matches: list of matching keypoint index pairs (as cv2.DMatch objects)

    Returns
    -------
        image_out: image1 and image2 joined side-by-side with matching lines; color image (BGR), uint8, values in [0, 255]
    """
    # Note: DO NOT use OpenCV's match drawing function(s)! Write your own :)
    image_out = np.float32(make_image_pair(image1, image2))
    image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    cv2.normalize(image_out,image_out,0,255,cv2.NORM_MINMAX)
    image_out = np.uint8(image_out)
    xAdj = image1.shape[1]
    hc = 0
    for m in matches:
        p1, p2 = kp1[m.queryIdx].pt, kp2[m.trainIdx].pt
        p1 = (int(round(p1[0])),int(round(p1[1])))
        p2 = (int(round(p2[0])+xAdj),int(round(p2[1])))
        cv2.line(image_out, p1, p2, HIGHLIGHT_CYCLE[hc], HIGHLIGHT_THICK, lineType=cv2.CV_AA)
        hc = (hc + 1) % len(HIGHLIGHT_CYCLE)
    return image_out

def compute_translation_RANSAC(kp1, kp2, matches, sigma=10):
    """Compute best translation vector using RANSAC given keypoint matches.

    Parameters
    ----------
        kp1: list of keypoints (cv2.KeyPoint objects) found in image1
        kp2: list of keypoints (cv2.KeyPoint objects) found in image2
        matches: list of matches (as cv2.DMatch objects)
        sigma: assumeds std of noise

    Returns
    -------
        translation: translation/offset vector <x, y>, NumPy array of shape (2, 1)
        good_matches: consensus set of matches that agree with this translation
    """
    # initialization
    loopNo, N = 0, sys.maxint
    pts1All = np.matrix([kp1[i.queryIdx].pt for i in matches])
    pts2All = np.matrix([kp2[i.trainIdx].pt for i in matches]) 
    s = 1             
    bestScore, good_matches_index, bestDiff = 0, [], []
    threshold = np.sqrt(3.84) * sigma
    
    # RANSAC
    while loopNo < N:
        # sample
        sample = np.random.choice(matches, s)
        pts1 = np.matrix([kp1[i.queryIdx].pt for i in sample])
        pts2 = np.matrix([kp2[i.trainIdx].pt for i in sample])
        # solve
        T = np.matrix([[1.0,0.0,pts2[0,0]-pts1[0,0]],
                       [0.0,1.0,pts2[0,1]-pts1[0,1]]], 
                       dtype=np.float_)
        # score
        inliers, diff = find_inliers(pts1All, pts2All, T, threshold)
        if len(inliers) > bestScore:
            bestScore = len(inliers)
            bestDiff = diff
            good_matches_index = inliers
            # recalculate N
            e = 1.0 - float(bestScore) / len(matches)
            N = np.log(1-0.99) / np.log(1 - (1 - e)**s)
            # print loopNo, e, N, bestScore
        loopNo += 1
    
    # good_matches
    good_matches = [matches[i] for i in good_matches_index]
    # compute for translation least mean squares (Ax = b)
    pts1gm = np.matrix([kp1[i.queryIdx].pt for i in good_matches])
    pts2gm = np.matrix([kp2[i.trainIdx].pt for i in good_matches])
    A = np.tile(np.eye(2, dtype=np.float_), (pts1gm.shape[0],1))
    b = (pts2gm - pts1gm).flatten().T
    x = np.linalg.lstsq(A, b)[0]
    translation = x
    return translation, good_matches
    
def compute_similarity_RANSAC(kp1, kp2, matches, sigma=10):
    """Compute best similarity transform using RANSAC given keypoint matches.

    Parameters
    ----------
        kp1: list of keypoints (cv2.KeyPoint objects) found in image1
        kp2: list of keypoints (cv2.KeyPoint objects) found in image2
        matches: list of matches (as cv2.DMatch objects)

    Returns
    -------
        transform: similarity transform matrix, NumPy array of shape (2, 3)
        good_matches: consensus set of matches that agree with this transform
    """

    # initialization
    loopNo, N = 0, sys.maxint
    pts1All = np.matrix([kp1[i.queryIdx].pt for i in matches])
    pts2All = np.matrix([kp2[i.trainIdx].pt for i in matches]) 
    s = 2
    bestScore, good_matches_index, bestDiff = 0, [], []
    threshold = np.sqrt(3.84) * sigma
    
    # RANSAC
    while loopNo < N:
        # sample
        sample = np.random.choice(matches, s)
        pts1 = np.matrix([kp1[i.queryIdx].pt for i in sample])
        pts2 = np.matrix([kp2[i.trainIdx].pt for i in sample])
        # solve
        A = np.matrix([[pts1[0,0],-pts1[0,1],1.0,0.0],
                       [pts1[0,1],pts1[0,0],0.0,1.0],
                       [pts1[1,0],-pts1[1,1],1.0,0.0],
                       [pts1[1,1],pts1[1,0],0.0,1.0]],
                       dtype=np.float_)
        b = pts2.flatten().T
        try:
            x = np.linalg.inv(A) * b
        except np.linalg.LinAlgError:
            continue
        T = np.matrix([[x[0,0],-x[1,0],x[2,0]],
                       [x[1,0],x[0,0],x[3,0]]], 
                       dtype=np.float_)
        # score
        inliers, diff = find_inliers(pts1All, pts2All, T, threshold)
        if len(inliers) > bestScore:
            bestScore = len(inliers)
            bestDiff = diff
            good_matches_index = inliers
            # recalculate N
            e = 1.0 - float(bestScore) / len(matches)
            N = np.log(1-0.99) / np.log(1 - (1 - e)**s)
            # print loopNo, e, N, bestScore
        loopNo += 1
    
    # good_matches
    good_matches = [matches[i] for i in good_matches_index]
    # compute for transform least mean squares (Ax = b)
    pts1gm = np.matrix([kp1[i.queryIdx].pt for i in good_matches])
    pts2gm = np.matrix([kp2[i.trainIdx].pt for i in good_matches])
    A = np.ndarray((0,4), dtype=np.float_)
    for i in xrange(pts1gm.shape[0]):
        u, v = pts1gm[i,0], pts1gm[i,1]
        A = np.append(A, [[u,-v,1.0,0.0],
                          [v,u,0.0,1.0]],
                      axis=0)
    A = np.matrix(A)
    b = pts2gm.flatten().T
    x = np.linalg.lstsq(A, b)[0]
    transform = np.matrix([[x[0,0],-x[1,0],x[2,0]],
                           [x[1,0],x[0,0],x[3,0]]], 
                           dtype=np.float_)
    return transform, good_matches

def compute_affine_RANSAC(kp1, kp2, matches, sigma=10):
    """Compute best affine transform using RANSAC given keypoint matches.

    Parameters
    ----------
        kp1: list of keypoints (cv2.KeyPoint objects) found in image1
        kp2: list of keypoints (cv2.KeyPoint objects) found in image2
        matches: list of matches (as cv2.DMatch objects)

    Returns
    -------
        transform: similarity transform matrix, NumPy array of shape (2, 3)
        good_matches: consensus set of matches that agree with this transform
    """

    # initialization
    loopNo, N = 0, sys.maxint
    pts1All = np.matrix([kp1[i.queryIdx].pt for i in matches])
    pts2All = np.matrix([kp2[i.trainIdx].pt for i in matches]) 
    s = 3
    bestScore, good_matches_index, bestDiff = 0, [], []
    threshold = np.sqrt(3.84) * sigma
    
    # RANSAC
    while loopNo < N:
        # sample
        sample = np.random.choice(matches, s)
        pts1 = np.matrix([kp1[i.queryIdx].pt for i in sample])
        pts2 = np.matrix([kp2[i.trainIdx].pt for i in sample])
        # solve
        A = np.matrix([[pts1[0,0],pts1[0,1],1.0,0.0,0.0,0.0],
                       [0.0,0.0,0.0,pts1[0,0],pts1[0,1],1.0],
                       [pts1[1,0],pts1[1,1],1.0,0.0,0.0,0.0],
                       [0.0,0.0,0.0,pts1[1,0],pts1[1,1],1.0],
                       [pts1[2,0],pts1[2,1],1.0,0.0,0.0,0.0],
                       [0.0,0.0,0.0,pts1[2,0],pts1[2,1],1.0]],
                       dtype=np.float_)
        b = pts2.flatten().T
        try:
            x = np.linalg.inv(A) * b
        except np.linalg.LinAlgError:
            continue
        T = np.matrix(x.reshape(2,3))
        # score
        inliers, diff = find_inliers(pts1All, pts2All, T, threshold)
        if len(inliers) > bestScore:
            bestScore = len(inliers)
            bestDiff = diff
            good_matches_index = inliers
            # recalculate N
            e = 1.0 - float(bestScore) / len(matches)
            N = np.log(1-0.99) / np.log(1 - (1 - e)**s)
            # print loopNo, e, N, bestScore
        loopNo += 1
    
    # good_matches
    good_matches = [matches[i] for i in good_matches_index]
    # compute for transform least mean squares (Ax = b)
    pts1gm = np.matrix([kp1[i.queryIdx].pt for i in good_matches])
    pts2gm = np.matrix([kp2[i.trainIdx].pt for i in good_matches])
    A = np.ndarray((0,6), dtype=np.float_)
    for i in xrange(pts1gm.shape[0]):
        u, v = pts1gm[i,0], pts1gm[i,1]
        A = np.append(A, [[u,v,1.0,0.0,0.0,0.0],
                          [0.0,0.0,0.0,u,v,1.0]],
                      axis=0)
    A = np.matrix(A)
    b = pts2gm.flatten().T
    x = np.linalg.lstsq(A, b)[0]
    transform = x.reshape(2,3)
    return transform, good_matches
    
def find_inliers(pts1, pts2, T, threshold):
    """Compute number of inliers

    Parameters
    ----------
        pts1: (N,2) points found in image1
        pts2: (N,2) points found in image2
        T: translation matrix
        threshold: threshold distance

    Returns
    -------
        inliers: index array of included points
        diff: array of difference
    """
    pts1in = np.concatenate((pts1,np.ones((pts1.shape[0],1))), axis=1)
    pts2out = (T * pts1in.T).T
    diff = np.sum(np.power(pts2 - pts2out, 2), axis=1)
    inliers = np.where(diff < threshold**2)[0]
    return [inliers[0,i] for i in xrange(inliers.size)], diff
    
def util1b(output, image):
    """Write image to file for #1b
    Parameters
    ----------
        output: output file
        image: input image
    """
    image2 = np.zeros(image.shape, dtype=image.dtype)
    cv2.normalize(image, image2, 0, 255, cv2.NORM_MINMAX)
    image2 = np.uint8(image2)
    cv2.imwrite(output, image2)
    return

def overlay_image(img1, img2):
    """overlay image1 and image2

    Parameters
    ----------
        img1: image1
        img2: image2

    Returns
    -------
        img_out: overlayed image where img1 in red, img2 in green uint8 [0,255]
    """
    img_out = np.zeros((img1.shape[0],img1.shape[1],3),
                       dtype=img1.dtype)
    img_out[:,:,2] = img1
    img_out[:,:,1] = img2
    return img_out

def warp_image(img, M):
    """warp image1 and image2

    Parameters
    ----------
        img: grayscale floating-point image1, values in [0.0, 1.0]
        M: affine transform (image2 = M * image1)

    Returns
    -------
        img_out: warped image
    """
    rows,cols = img.shape
    iM = cv2.invertAffineTransform(M)
    img_out = cv2.warpAffine(img, iM, (cols,rows))
    return img_out
    
# Driver code
def main():
    np.random.seed(87)
    # Note: Comment out parts of this code as necessary
    # 1a
    transA = cv2.imread(os.path.join(input_dir, "transA.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float_) / 255.0
    transA_Ix = gradientX(transA)
    transA_Iy = gradientY(transA)
    transA_pair = make_image_pair(transA_Ix, transA_Iy)
    cv2.normalize(transA_pair, transA_pair, 0, 255, cv2.NORM_MINMAX)
    transA_pair = np.uint8(transA_pair)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-a-1.png"), transA_pair)
    
    # Similarly for simA.jpg
    simA = cv2.imread(os.path.join(input_dir, "simA.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float_) / 255.0
    simA_Ix = gradientX(simA)
    simA_Iy = gradientY(simA)
    simA_pair = make_image_pair(simA_Ix, simA_Iy)
    cv2.normalize(simA_pair, simA_pair, 0, 255, cv2.NORM_MINMAX)
    simA_pair = np.uint8(simA_pair)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-a-2.png"), simA_pair)
    
    transB = cv2.imread(os.path.join(input_dir, "transB.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float_) / 255.0
    transB_Ix = gradientX(transB)
    transB_Iy = gradientY(transB)
    simB = cv2.imread(os.path.join(input_dir, "simB.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float_) / 255.0
    simB_Ix = gradientX(simB)
    simB_Iy = gradientY(simB)
    
    # 1b
    kernel = cv2.getGaussianKernel(5,1)
    kernel = kernel * kernel.T
    alpha = 0.04
    #kernel = np.ones((3, 3), dtype=np.float_) / 9.0
    transA_R = harris_response(transA_Ix, transA_Iy, kernel, alpha)
    util1b(os.path.join(output_dir, "ps5-1-b-1.png"), transA_R)
    
    transB_R = harris_response(transB_Ix, transB_Iy, kernel, alpha)
    util1b(os.path.join(output_dir, "ps5-1-b-2.png"), transB_R)
    
    simA_R = harris_response(simA_Ix, simA_Iy, kernel, alpha)
    util1b(os.path.join(output_dir, "ps5-1-b-3.png"), simA_R)
    
    simB_R = harris_response(simB_Ix, simB_Iy, kernel, alpha)
    util1b(os.path.join(output_dir, "ps5-1-b-4.png"), simB_R)
    
    # 1c
    radius = 9.5
    transA_corners = find_corners(transA_R, 0.001, radius)
    print "transA: max={0:0.3f} min={1:0.3f} corners={2}".format(transA_R.max(), transA_R.min(), len(transA_corners))
    transA_out = draw_corners(transA, transA_corners)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-c-1.png"), transA_out)
    
    transB_corners = find_corners(transB_R, 0.001, radius)
    print "transB: max={0:0.3f} min={1:0.3f} corners={2}".format(transB_R.max(), transB_R.min(), len(transB_corners))
    transB_out = draw_corners(transB, transB_corners)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-c-2.png"), transB_out)
    
    simA_corners = find_corners(simA_R, 0.001, radius)
    print "simA: max={0:0.3f} min={1:0.3f} corners={2}".format(simA_R.max(), simA_R.min(), len(simA_corners))
    simA_out = draw_corners(simA, simA_corners)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-c-3.png"), simA_out)
    
    simB_corners = find_corners(simB_R, 0.001, radius)
    print "simB: max={0:0.3f} min={1:0.3f} corners={2}".format(simB_R.max(), simB_R.min(), len(simB_corners))
    simB_out = draw_corners(simB, simB_corners)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-c-4.png"), simB_out)
    
    # 2a
    _size = radius * 2.0
    # (transA, transB) pair
    transA_angle = gradient_angle(transA_Ix, transA_Iy)
    transA_kp = get_keypoints(transA_corners, transA_R, transA_angle, _size=_size, _octave=0)
    transA_out = draw_keypoints(transA, transA_kp)
    transB_angle = gradient_angle(transB_Ix, transB_Iy)
    transB_kp = get_keypoints(transB_corners, transB_R, transB_angle, _size=_size, _octave=0)
    transB_out = draw_keypoints(transB, transB_kp)
    trans_pair = make_image_pair(transA_out, transB_out)
    cv2.imwrite(os.path.join(output_dir, "ps5-2-a-1.png"), trans_pair)
    
    # (simA, simB) pair
    simA_angle = gradient_angle(simA_Ix, simA_Iy)
    simA_kp = get_keypoints(simA_corners, simA_R, simA_angle, _size=21.0, _octave=0)
    simA_out = draw_keypoints(simA, simA_kp)
    simB_angle = gradient_angle(simB_Ix, simB_Iy)
    simB_kp = get_keypoints(simB_corners, simB_R, simB_angle, _size=21.0, _octave=0)
    simB_out = draw_keypoints(simB, simB_kp)
    sim_pair = make_image_pair(simA_out, simB_out)
    cv2.imwrite(os.path.join(output_dir, "ps5-2-a-2.png"), sim_pair)
    
    # 2b
    transA_desc = get_descriptors(transA, transA_kp)
    transB_desc = get_descriptors(transB, transB_kp)
    trans_matches = match_descriptors(transA_desc, transB_desc)
    # Draw matches and write to file:
    trans_pair = draw_matches(transA, transB, transA_kp, transB_kp, trans_matches)
    cv2.imwrite(os.path.join(output_dir, "ps5-2-b-1.png"), trans_pair)

    simA_desc = get_descriptors(simA, simA_kp)
    simB_desc = get_descriptors(simB, simB_kp)
    sim_matches = match_descriptors(simA_desc, simB_desc)
    # Draw matches and write to file:
    sim_pair = draw_matches(simA, simB, simA_kp, simB_kp, sim_matches)
    cv2.imwrite(os.path.join(output_dir, "ps5-2-b-2.png"), sim_pair)

    # 3a
    # Compute translation vector using RANSAC for (transA, transB) pair, draw biggest consensus set
    trans_T, trans_gm = compute_translation_RANSAC(transA_kp, transB_kp, trans_matches, sigma=3.0)
    print "3a:"
    print trans_T
    print "match = {0}/{1} or {2:0.2f}%".format(len(trans_gm), len(trans_matches), 100.0 * len(trans_gm) / len(trans_matches))
    trans_pair = draw_matches(transA, transB, transA_kp, transB_kp, trans_gm)
    cv2.imwrite(os.path.join(output_dir, "ps5-3-a-1.png"), trans_pair)
    
    # 3b
    # Compute similarity transform for (simA, simB) pair, draw biggest consensus set
    sim_S, sim_gm = compute_similarity_RANSAC(simA_kp, simB_kp, sim_matches, sigma=2.0)
    print "3b:"
    print sim_S
    print "match = {0}/{1} or {2:0.2f}%".format(len(sim_gm), len(sim_matches), 100.0 * len(sim_gm) / len(sim_matches))
    sim_pair = draw_matches(simA, simB, simA_kp, simB_kp, sim_gm)
    cv2.imwrite(os.path.join(output_dir, "ps5-3-b-1.png"), sim_pair)
    
    # 3c
    # Compute affine transform for (simA, simB) pair, draw biggest consensus set
    sim_A, sim_gm = compute_affine_RANSAC(simA_kp, simB_kp, sim_matches, sigma=2.0)
    print "3c:"
    print sim_A
    print "match = {0}/{1} or {2:0.2f}%".format(len(sim_gm), len(sim_matches), 100.0 * len(sim_gm) / len(sim_matches))
    sim_pair = draw_matches(simA, simB, simA_kp, simB_kp, sim_gm)
    cv2.imwrite(os.path.join(output_dir, "ps5-3-c-1.png"), sim_pair)

    # 3d
    warpedB = warp_image(simB, sim_S)
    util1b(os.path.join(output_dir, "ps5-3-d-1.png"), warpedB)
    sim_overlay = overlay_image(simA, warpedB)
    util1b(os.path.join(output_dir, "ps5-3-d-2.png"), sim_overlay)
    
    # 3e
    warpedB = warp_image(simB, sim_A)
    util1b(os.path.join(output_dir, "ps5-3-e-1.png"), warpedB)
    sim_overlay = overlay_image(simA, warpedB)
    util1b(os.path.join(output_dir, "ps5-3-e-2.png"), sim_overlay)
    
if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
