"""Problem Set 4: Geometry."""

import numpy as np
import cv2

import os
import sys

# I/O directories
input_dir = "input"
output_dir = "output"

# Input files
PIC_A = "pic_a.jpg"
PIC_A_2D = "pts2d-pic_a.txt"
PIC_A_2D_NORM = "pts2d-norm-pic_a.txt"
PIC_B = "pic_b.jpg"
PIC_B_2D = "pts2d-pic_b.txt"
SCENE = "pts3d.txt"
SCENE_NORM = "pts3d-norm.txt"

# Utility code
def read_points(filename):
    """Read point data from given file and return as NumPy array."""
    with open(filename) as f:
        lines = f.readlines()
        pts = []
        for line in lines:
            pts.append(map(float, line.split()))
    return np.array(pts)


# Assignment code
def solve_least_squares(pts3d, pts2d):
    """Solve for transformation matrix M that maps each 3D point to corresponding 2D point using the least squares method.

    Parameters
    ----------
        pts3d: 3D (object) points, NumPy array of shape (N, 3)
        pts2d: corresponding 2D (image) points, NumPy array of shape (N, 2)

    Returns
    -------
        M: transformation matrix, NumPy array of shape (3, 4)
        error: sum of squared residuals of all points
    """
    # construct A
    A = np.ndarray((0,12), dtype=pts3d.dtype);
    for i in xrange(pts3d.shape[0]):
        x, y, z = pts3d[i,0], pts3d[i,1], pts3d[i,2]
        u, v = pts2d[i,0], pts2d[i,1]
        A = np.append(A, [[x,y,z,1,0,0,0,0,-u*x,-u*y,-u*z,-u],
                          [0,0,0,0,x,y,z,1,-v*x,-v*y,-v*z,-v]],
                      axis=0)
    A = np.matrix(A)
    # compute SVD
    U, s, V = np.linalg.svd(A, full_matrices=True)
    # M = V eigenvector with smallest eigenvalue (last row of V)
    M = V[-1,:].reshape(V.shape[1],1)
    # compute error
    error = (A*M)
    error = (error.T * error)[0,0]
    # reshape M
    M = M.reshape(3,4)
    
    return M, error


def project_points(pts3d, M):
    """Project each 3D point to 2D using matrix M.

    Parameters
    ----------
        pts3d: 3D (object) points, NumPy array of shape (N, 3)
        M: projection matrix, NumPy array of shape (3, 4)

    Returns
    -------
        pts2d_projected: projected 2D points, NumPy array of shape (N, 2)
    """
    # construct 3d points homogenous matrix
    pts3dH = np.append(pts3d,np.ones((pts3d.shape[0],1)),axis=1)
    pts3dH = np.matrix(pts3dH.T)
    # calculate for 2d points homogenous matrix
    pts2dH = M * pts3dH
    # convert 2d points homogenous matrix to actual 2d points
    pts2d_projected = np.array(pts2dH[0:2,:].T)
    pts2d_s = np.array(pts2dH[2:3,:].T)
    pts2d_projected = pts2d_projected / pts2d_s

    return pts2d_projected    


def get_residuals(pts2d, pts2d_projected):
    """Compute residual error for each point.

    Parameters
    ----------
        pts2d: observed 2D (image) points, NumPy array of shape (N, 2)
        pts2d_projected: 3D (object) points projected to 2D, NumPy array of shape (N, 2)

    Returns
    -------
        residuals: residual error for each point (L2 distance between observed and projected 2D points)
    """
    residuals = np.sqrt(np.sum((pts2d - pts2d_projected)**2,axis=1))
    return residuals


def calibrate_camera(pts3d, pts2d):
    """Find the best camera projection matrix given corresponding 3D and 2D points.

    Parameters
    ----------
        pts3d: 3D (object) points, NumPy array of shape (N, 3)
        pts2d: corresponding 2D (image) points, NumPy array of shape (N, 2)

    Returns
    -------
        bestM: best transformation matrix, NumPy array of shape (3, 4)
        error: sum of squared residuals of all points for bestM
    """
    bestM = None
    bestError = sys.float_info.max
    for i in xrange(1,11):
        ptsIdx = np.arange(0,pts3d.shape[0])
        np.random.shuffle(ptsIdx)
        print "i={0}".format(i),
        for k in [8,12,16]:
            M, error = solve_least_squares(pts3d[ptsIdx[:k],:], pts2d[ptsIdx[:k],:])
            pts2d_projected = project_points(pts3d[ptsIdx[-4:],:], M)
            residuals = get_residuals(pts2d[ptsIdx[-4:],:], pts2d_projected)
            if error < bestError:
                bestM = M
                bestError = error
            print np.mean(residuals),
        print        
    return bestM, bestError


def compute_fundamental_matrix(pts2d_a, pts2d_b):
    """Compute fundamental matrix given corresponding points from 2 images of a scene.

    Parameters
    ----------
        pts2d_a: 2D points from image A, NumPy array of shape (N, 2)
        pts2d_b: corresponding 2D points from image B, NumPy array of shape (N, 2)

    Returns
    -------
        F: the fundamental matrix
    """

    # construct A
    A = np.ndarray((0,9), dtype=pts2d_a.dtype);
    for i in xrange(pts2d_a.shape[0]):
        u, v = pts2d_a[i,0], pts2d_a[i,1]
        u2, v2 = pts2d_b[i,0], pts2d_b[i,1]
        A = np.append(A, [[u2*u,u2*v,u2,v2*u,v2*v,v2,u,v,1.0]],
                      axis=0)
    A = np.matrix(A)
    # compute SVD
    U, s, V = np.linalg.svd(A, full_matrices=True)
    # M = V eigenvector with smallest eigenvalue (last row of V)
    F = V[-1,:].reshape(V.shape[1],1)
    # compute error
    error = (A*F)
    error = (error.T * error)[0,0]
    # reshape F
    F = F.reshape(3,3)
    return F

def compute_norm_matrix(pts2d):
    cx = pts2d[:,0].mean()
    cy = pts2d[:,1].mean()
    s1 = max(pts2d[:,0].max()-cx,cx-pts2d[:,0].min())
    s2 = max(pts2d[:,1].max()-cy,cy-pts2d[:,1].min())
    s = 1.0 / max(s1,s2)
    #s = 1.0 / pts2d[:,:].std()
    #print s, cx, cy
    T = np.matrix([[s,0.0,0.0],[0.0,s,0.0],[0.0,0.0,1.0]]) * np.matrix([[1.0,0.0,-cx],[0.0,1.0,-cy],[0.0,0.0,1.0]])
    return T
    
# Driver code
def main():
    """Driver code."""
    np.random.seed(125)
    # 1a
    # Read points
    pts3d_norm = read_points(os.path.join(input_dir, SCENE_NORM))
    pts2d_norm_pic_a = read_points(os.path.join(input_dir, PIC_A_2D_NORM))

    # Solve for transformation matrix using least squares
    M, error = solve_least_squares(pts3d_norm, pts2d_norm_pic_a)

    # Project 3D points to 2D
    pts2d_projected = project_points(pts3d_norm, M)

    # Compute residual error for each point
    residuals = get_residuals(pts2d_norm_pic_a, pts2d_projected)

    # Print the <u, v> projection of the last point, and the corresponding residual
    print "ps4-1a:"
    print "M="
    print M
    print "<u,v>="
    print "orig", pts2d_norm_pic_a[-1,:], "projected", pts2d_projected[-1,:]
    print "residuals="
    print residuals[-1]

    # 1b
    # Read points
    pts3d = read_points(os.path.join(input_dir, SCENE))
    pts2d_pic_b = read_points(os.path.join(input_dir, PIC_B_2D))
    # NOTE: These points are not normalized

    # Use the functions from 1a to implement calibrate_camera() and find the best transform (bestM)
    print "ps4-1b:"
    bestM, error = calibrate_camera(pts3d, pts2d_pic_b)
    print "bestM="
    print bestM
    
    # 1c
    # Compute the camera location using bestM
    C = -np.linalg.inv(bestM[:,0:3])*bestM[:,3:4]
    print "ps4-1c:"
    print "center", C.T

    # 2a
    # Implement compute_fundamental_matrix() to find the raw fundamental matrix
    pts2d_pic_a = read_points(os.path.join(input_dir, PIC_A_2D))
    pts2d_pic_b = read_points(os.path.join(input_dir, PIC_B_2D))
    F = compute_fundamental_matrix(pts2d_pic_a, pts2d_pic_b)
    #t1 = np.matrix(np.append(pts2d_pic_a[0,:],[[1.0]])).T
    #t2 = np.matrix(np.append(pts2d_pic_b[0,:],[[1.0]])).T
    #print t1, t2
    #print t2.T*F*t1
    print "ps4-2a:"
    print "F~="
    print F
    
    # 2b
    # Reduce the rank of the fundamental matrix
    U, s, V = np.linalg.svd(F, full_matrices=True)
    s2 = s.copy()
    s2[-1] = 0.0
    F2 = U*np.diag(s2)*V
    F2T = F2.T
    #print t2.T*F2*t1
    print "ps4-2b:"
    print "F="
    print F2
    
    # 2e
    Ta = compute_norm_matrix(pts2d_pic_a)
    Tb = compute_norm_matrix(pts2d_pic_b)
    pts2d_norm_pic_a = project_points(pts2d_pic_a, Ta)
    pts2d_norm_pic_b = project_points(pts2d_pic_b, Tb)
    Fnorm = compute_fundamental_matrix(pts2d_norm_pic_a, pts2d_norm_pic_b)
    Unorm, snorm, Vnorm = np.linalg.svd(Fnorm, full_matrices=True)
    snorm2 = snorm.copy()
    snorm2[-1] = 0.0
    Fnorm2 = Unorm*np.diag(snorm2)*Vnorm
    
    print "ps4-2d:"
    print "Ta="
    print Ta
    print "Tb="
    print Tb
    print "Fhat="
    print Fnorm2
    #print pts2d_norm_pic_a
    #print pts2d_norm_pic_b
    
    # 2c and 2e
    # Draw epipolar lines
    # load images
    pic_a = cv2.imread(os.path.join(input_dir, PIC_A),-1)
    pic_b = cv2.imread(os.path.join(input_dir, PIC_B),-1)
    pic_a2 = pic_a.copy()
    pic_b2 = pic_b.copy()
    # prepare border lines
    lineL = np.cross(np.matrix([0.0,0.0,1.0]),
                     np.matrix([0.0,pic_a.shape[0]-1.0,1.0]))
    lineR = np.cross(np.matrix([pic_a.shape[1]-1.0,0.0,1.0]),
                     np.matrix([pic_a.shape[1]-1.0,pic_a.shape[0]-1.0,1.0]))
    # plot epipolar lines
    Fnorm3 = Tb.T*Fnorm2*Ta
    Fnorm3T = Fnorm3.T
    iterSet = [("pic_a", pic_a, F2T, pts2d_pic_b),
               ("pic_b", pic_b, F2,  pts2d_pic_a),
               ("pic_a2", pic_a2, Fnorm3T, pts2d_pic_b),
               ("pic_b2", pic_b2, Fnorm3,  pts2d_pic_a)]
    for title, img, Fmat, pts2d in iterSet:
        for pts in pts2d:
            ptsI = np.matrix(np.append(pts,[1.0])).T
            lineI = (Fmat*ptsI).T
            ptsL = np.cross(lineI, lineL)
            ptsR = np.cross(lineI, lineR)
            if ptsL[0,2] == 0:
                print "pass", ptsL
                continue
            else:
                ptsL = tuple(np.int32((ptsL[0,0:2] / ptsL[0,2]).round()))
            if ptsR[0,2] == 0:
                print "pass", ptsR
                continue
            else:
                ptsR = tuple(np.int32((ptsR[0,0:2] / ptsR[0,2]).round()))
            cv2.line(img, ptsL, ptsR, (255,0,0), 1)
    # output images
    print "ps4-2e:"
    print "Fnew="
    print Fnorm3
    cv2.imwrite(os.path.join(output_dir, 'ps4-2-c-1.png'), pic_a)
    cv2.imwrite(os.path.join(output_dir, 'ps4-2-c-2.png'), pic_b)
    cv2.imwrite(os.path.join(output_dir, 'ps4-2-e-1.png'), pic_a2)
    cv2.imwrite(os.path.join(output_dir, 'ps4-2-e-2.png'), pic_b2)

if __name__ == '__main__':
    main()
