"""
CS8803 AI for Robotics Final Project
Written By: Luke Wicent Sy (gtid: lsy3 gtid#:903184380)
"""

import sys
import os
import pickle
import numpy as np
from scipy.ndimage.filters import gaussian_filter
# from scipy.cluster.vq import kmeans

DEBUG = False
# Kalman Filter Parameters
dt = 1.0

# Trajectory Parameters
SGMNT_LEN = 10 # actual trajectory length is 2 times this
#STATE_NUM = -1 # if set to -1, does not do KMEANS
               # if set to a numder, does KMEANS on the sample data

# Border Parameters
ALLOWANCE = 5 # pixel allowance for out-of-bounds check in KF border check
BOUNDS = [558-ALLOWANCE, 82+ALLOWANCE, 324-ALLOWANCE, 41+ALLOWANCE] # right, left, down, up border in KF
ALLOWANCE2 = 0 # pixel allowance for out-of-bounds check in trajectory matching
BOUNDS2 = [558+ALLOWANCE2, 82-ALLOWANCE2, 324+ALLOWANCE2, 41-ALLOWANCE2] # right, left, down, up border in trajectory matching
CBOUNDS = [(558+82)/2.0, (324+41)/2.0, 35**2] # x_center, y_center, radius

                   
class Hexbug:
    def __init__(self, init_x):
        """
        Hexbug class initialization 
        """
        
        # x initialization
        self.x = np.concatenate((init_x, np.array([0.0, 0.0])), axis=1)
        self.x = np.matrix(self.x).T
        
        # note: x = A*x + B*U + W
        self.A = np.matrix([[1.0, 0.0, dt, 0.0],
                           [0.0, 1.0, 0.0, dt],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
        self.B = np.matrix([[dt**2/2],[dt**2/2], [dt], [dt]])
        self.U = 0.0
        self.W = np.matrix([[0.0], [0.0], [0.0], [0.0]])
        
        # covariance matrix
        self.Ex = np.matrix([[dt**4/4, 0, dt**3/2, 0],
                             [0, dt**4/4, 0, dt**3/2],
                             [dt**3/2, 0, dt**2, 0],
                             [0, dt**3/2, 0, dt**2]]) * (.1**2)
        self.P = self.Ex.copy()
        self.Porig = self.Ex.copy() * 100
        
        # observation matrix
        self.H = np.matrix([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
        
        # Ez
        self.Ez = np.matrix([[0.1, 0.0, 0.0, 0.0],
                            [0.0, 0.1, 0.0, 0.0],
                            [0.0, 0.0, 0.1, 0.0],
                            [0.0, 0.0, 0.0, 0.1]])
        
        # identity matrix
        self.I = np.matrix(np.identity(4))
        
        # past observed data
        self.o_past = np.matrix(init_x).T
               
    def observe(self, o):
        """
        Learn and predict next state using Kalman Filter
        Args:
            o: observed measurements
        """
        
        self.x = self.A*self.x + self.B*self.U + self.W
        self.P = self.A*self.P*self.A.T + self.Ex
        
        S = self.H*self.P*self.H.T + self.Ez
        K = self.P*self.H.T*S.getI()
        
        o = np.matrix(o).T
        y = np.concatenate((o, (o-self.o_past)/dt), axis=0) #+ Z
        self.x = self.x + K*(y-self.H*self.x)
        self.P = (self.I-K*self.H)*self.P
        
        # check if next estimated state is out of bounds and do the necessary updates
        new_x, change = self.border_update(self.x)
        if change["code"] > 0:
            self.x = new_x
            self.P = self.Porig.copy()
            
        self.o_past = o
        
        return self.x.copy(), change
    
    def predictOne(self):
        """
        Predict next state (one step only)
        """
        
        if DEBUG: print "vel", np.sqrt(np.power(self.x[2:],2).sum()), self.x[2:].T
        # check velocity. if velocity is very low, assume hexbug is not moving
        if np.sqrt(np.power(self.x[2:],2).sum()) <= 3.5:
            self.x[2:] = 0.0
        
        # update state (x)
        self.x = self.A*self.x
        new_x, change = self.border_update(self.x)
        if change["code"] > 0:
            self.x = new_x
            
        return self.x, change
        
    def predict(self, steps):
        """
        Predict next state (multiple steps)
        """
        
        out = []
        
        x = self.x.copy()
        # update starting x with the last observed measurement
        x[:2] = self.o_past
        # check velocity. if velocity is very low, assume hexbug is not moving
        if np.sqrt(np.power(x[2:],2).sum()) <= 2.0:
            x[2:] = 0.0
        
        # start prediction
        for i in xrange(steps):
            x = self.A*x
            
            new_x, change = self.border_update(x)
            if change["code"] > 0:
                x = new_x
                
            out.append(x[0:2])
        return out

    def border_update(self, x):
        """
        Border check
        Args:
            x: state (4, 1) array
        """
        
        change = {"code": 0}
        xcpy = x.copy()
        
        if xcpy[0][0] >= BOUNDS[0] and xcpy[2][0] > 0: #right
            xcpy[0][0] = (2*BOUNDS[0]-xcpy[0][0])
            xcpy[2][0] *= -1
            change["code"] = 1
        elif xcpy[0][0] <= BOUNDS[1] and xcpy[2][0] < 0: #left
            xcpy[0][0] = (2*BOUNDS[1]-xcpy[0][0])
            xcpy[2][0] *= -1
            change["code"] = 2
        elif xcpy[1][0] >= BOUNDS[2] and xcpy[3][0] > 0: #down
            xcpy[1][0] = (2*BOUNDS[2]-xcpy[1][0])
            xcpy[3][0] *= -1
            change["code"] = 3
        elif xcpy[1][0] <= BOUNDS[3] and xcpy[3][0] < 0: #up
            xcpy[1][0] = (2*BOUNDS[3]-xcpy[1][0])
            xcpy[3][0] *= -1
            change["code"] = 4
        elif np.abs((xcpy[0][0]-CBOUNDS[0])**2+(xcpy[1][0]-CBOUNDS[1])**2) < CBOUNDS[2]:
            b_x0, b_y0 = (xcpy[0][0] - xcpy[2][0]).item(0), (xcpy[1][0] - xcpy[3][0]).item(0)
            b_x1, b_y1 = b_x0, b_y0
            for i in xrange(100):
                if ((xcpy[0][0] - i*xcpy[2][0]/100.0 - CBOUNDS[0])**2 + (xcpy[1][0] - i*xcpy[3][0]/100.0 - CBOUNDS[1])**2) > CBOUNDS[2]:
                    b_x1 = (xcpy[0][0] - i*xcpy[2][0]/100.0).item(0)
                    b_y1 = (xcpy[1][0] - i*xcpy[3][0]/100.0).item(0)
                    break
            v1 = np.array([b_x1-b_x0, b_y1-b_y0])
            v2 = np.array([-b_x1+CBOUNDS[0], -b_y1+CBOUNDS[1]])
            v2 = v2 / np.sqrt((v2**2).sum())
            v3 = np.dot(v1, v2) * v2
            v4 = 2*(v1-v3) + np.array([b_x0, b_y0]) - np.array([b_x1, b_y1])
            if v4.sum() == 0:
                v4 = np.array([0.0, 0.0])
            else:
                v4 = v4 / np.sqrt((v4**2).sum()) * np.sqrt(xcpy[2][0]**2+xcpy[3][0]**2).item(0)
            xcpy[2][0] = v4[0]
            xcpy[3][0] = v4[1]
            change["code"] = 5
            change["intersect"] = np.array([b_x1, b_y1])
        
        return xcpy, change

class HexbugPredictor:
    def __init__(self):
        """
        Initialize HexbugPredictor class
        """
        pass
        
    def learn_trajectory_segments(self, data):
        """
        Learn the base trajectory segments
        Args:
            data: (N, 2) array of coordinates
        """
        
        self.sample = []
        self.sample_type = []
        hb = None
        hkdist = np.sqrt(np.power(data[:,0]-CBOUNDS[0],2) + np.power(data[:,1]-CBOUNDS[1],2))
        
        for i in xrange(SGMNT_LEN, data.shape[0]-SGMNT_LEN):
            window_x = data[i-SGMNT_LEN:i+SGMNT_LEN,0]
            window_y = data[i-SGMNT_LEN:i+SGMNT_LEN,1]

            if window_x.argmax() == SGMNT_LEN-1: # right
                dbuf, theta = self.normalize(data[i-SGMNT_LEN:i+SGMNT_LEN,:], {"code":1})
                self.sample.append(dbuf.flatten())
                self.sample_type.append(1)
            if window_x.argmin() == SGMNT_LEN-1: # left
                dbuf, theta = self.normalize(data[i-SGMNT_LEN:i+SGMNT_LEN,:], {"code":2})
                self.sample.append(dbuf.flatten())
                self.sample_type.append(2)
            if window_y.argmax() == SGMNT_LEN-1: # down
                dbuf, theta = self.normalize(data[i-SGMNT_LEN:i+SGMNT_LEN,:], {"code":3})
                self.sample.append(dbuf.flatten())
                self.sample_type.append(3)
            if window_y.argmin() == SGMNT_LEN-1: # up
                dbuf, theta = self.normalize(data[i-SGMNT_LEN:i+SGMNT_LEN,:], {"code":4})
                self.sample.append(dbuf.flatten())
                self.sample_type.append(4)
            if hkdist[i-SGMNT_LEN:i+SGMNT_LEN].argmin() == SGMNT_LEN-1: # center
                dbuf, theta = self.normalize(data[i-SGMNT_LEN:i+SGMNT_LEN,:], {"code":5})
                self.sample.append(dbuf.flatten())
                self.sample_type.append(5)
                
        self.sample = np.array(self.sample)
        self.sample_type = np.array(self.sample_type)
        
        """
        if STATE_NUM > 0:
            sample = self.sample[self.sample_type==5]
            sample2, noise = kmeans(self.sample[self.sample_type!=5], STATE_NUM)
            self.sample_type = np.ones((sample.shape[0]+sample2.shape[0]))
            self.sample_type[:sample.size] = 5
            self.sample_type[sample.size:] = 1
            self.sample = np.array(np.vstack((sample, sample2)))
        """
        
        if DEBUG:
            print "sample"                    
            print self.sample.shape
            for i in xrange(self.sample.shape[0]):
                print i, self.sample_type[i], self.sample[i,:]
                
    def normalize(self, data, change):
        """
        Normalize trajectory (set origin to 0,0. orientation)
        Args:
            data: (N, 2) array of coordinates
            change: includes information on trajectory (i.e.: which wall it hit)
        Returns:
            normalized data
            angle: change in orientation
        """
             
        newdata = data.copy()
       
        # set start loc to (0,0)
        newdata -= newdata[0,:]
        
        # change orientation
        if change["code"] == 1:
            theta = np.pi / 2
        elif change["code"] == 2:
            theta = -np.pi / 2
        elif change["code"] == 3:
            theta = np.pi
        elif change["code"] == 4:
            theta = 0
        else:
            theta = 0
        
        rmat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        for i in xrange(newdata.shape[0]):
            newdata[i,:] = np.dot(newdata[i,:], rmat)
        
        return newdata, theta
    
    def find_trajectory(self, data, change):
        """
        Find the most similar trajectory from base trajectories
        Args:
            data: input trajectory. (N, 2) array of coordinates
            change: includes information on trajectory (i.e.: which wall it hit)
        Returns:
            matching base trajectory. (N, 2) array of coordinates.
            realigned trajectory computed from input trajectory and base trajectory. (N, 2) array of coordinates.
        """
        # normalize input trajectory
        dataorig = data.copy()
        data, theta = self.normalize(data, change)
        data = data.flatten()
        
        # compute trajectory distances
        mag = (np.power(self.sample[:,:data.size] - data, 2)).sum(axis=1)
        
        # computation / preparation to remove trajectories that leads to outside border
        sample2 = self.sample.copy()
        
        # adjust orientation
        theta2 = 2*np.pi - theta
        rmat = np.array([[np.cos(theta2), -np.sin(theta2)],[np.sin(theta2), np.cos(theta2)]])
        for i in xrange(0,sample2.shape[1],2):
            sample2[:,i:i+2] = np.dot(sample2[:,i:i+2], rmat)
        
        # adjust origin
        sample2[:,::2] = sample2[:,::2] - sample2[:,dataorig.size-2].reshape(-1,1) + dataorig[-1,0].item(0)
        sample2[:,1::2] = sample2[:,1::2] - sample2[:,dataorig.size-1].reshape(-1,1) + dataorig[-1,1].item(0)
        
        # finalize array that indicates which went out of bounds
        outside = ((sample2[:,::2] > BOUNDS2[0]).any(axis=1) | (sample2[:,::2] < BOUNDS2[1]).any(axis=1) | 
                   (sample2[:,1::2] > BOUNDS2[2]).any(axis=1) | (sample2[:,1::2] < BOUNDS2[3]).any(axis=1))
        
        # filter out the trajectories that went out of bounds and of the different type (which border it hit)
        if change["code"] >= 1 and change["code"] <= 4:
            mag = mag + mag.max() * ((self.sample_type < 1) | (self.sample_type > 4) | outside).reshape(-1,1)
        else:
            mag = mag + mag.max() * (self.sample_type != change["code"] | outside).reshape(-1,1)
                
        if DEBUG:
            print "id", mag.argmin(), mag[mag.argmin()], outside[mag.argmin()]
            print "sample", self.sample[mag.argmin(),:].tolist()
            print "sample2", sample2[mag.argmin(),:].tolist()
            print "data", data

        base_traj = self.sample[mag.argmin(),:].copy().reshape(-1,2)        
        realigned = sample2[mag.argmin(),:].copy().reshape(-1,2)
        """
        # clip out of bounds
        realigned[:,0] = np.clip(realigned[:,0], BOUNDS[1], BOUNDS[0])
        realigned[:,1] = np.clip(realigned[:,1], BOUNDS[3], BOUNDS[2])
        """
        
        return base_traj, realigned
                     
    def predict(self, data, steps=60):
        """
        Predict next states
        Args:
            data: input data. (N, 2) array of coordinates.
            steps: number of states to predict in the future
        Returns:
            out: list of (x, y) coordinates
        """
        
        # Learn from known data using KF while taking note of segments (partitioning it every time it hits the wall)
        hb = None
        hb_datacnt = 0
        hb_di = -1
        change = {"code":0}
        
        for i in xrange(data.shape[0]):
            if hb == None:
                hb = Hexbug(data[i,:])
            else:
                x, change = hb.observe(data[i,:])
                if change["code"] > 0 and (i-SGMNT_LEN)>=0 and (i+SGMNT_LEN)<=data.shape[0]:
                    hb = None
                    hb_datacnt = -1
                    hb_di = i
            hb_datacnt += 1
        
        # Data preparation after going through the known states
        if hb_datacnt < SGMNT_LEN:
            # if the segment after hitting the wall is too small
            change["code"] = 1
            last_segment = data[hb_di-SGMNT_LEN+1:].copy()
        else:
            # normal case
            change["code"] = 0
            hb.x[:2] = data[-1,:2].reshape(-1,1)
            last_segment = data[hb_di:].copy()
            last_segment = last_segment[-SGMNT_LEN:]
            
        if DEBUG:
            print "hb_datacnt", hb_datacnt, hb.x.T, last_segment.tolist()
    
        out = []
        while len(out) < steps:
            if change["code"] != 0:
                # hexbug movement if a wall was hit

                # match best fit with last_segment
                traj, realigned_traj = self.find_trajectory(last_segment, change)
                out.extend(realigned_traj[last_segment.shape[0]:,:].tolist())
                
                # realigning last segment
                last_segment = realigned_traj[-SGMNT_LEN:,:].copy()
                
                
                # make a new hexbug and train it using the outputted segment
                hb = Hexbug(np.array(last_segment[0,:]).flatten())
                for i in xrange(1,last_segment.shape[0]):
                    hb.observe(last_segment[i,:])
                hb.x[:2] = last_segment[-1,:2].reshape(-1,1)
                change["code"] = 0

                if DEBUG:
                    print "normalized segment", self.normalize(last_segment, change)[0].tolist()
                    print "kmeans traj", change, traj.tolist()
                    print "kmeans hidden", realigned_traj[:last_segment.shape[0],:].tolist()
                    print "kmeans realigned", last_segment.shape[0], realigned_traj[last_segment.shape[0]:,:].tolist()
                    print "last_segment", last_segment.tolist()
            else:
                # normal hexbug movement (no wall hitting)
                # state prediction with KF
                x, change = hb.predictOne()
                last_segment = np.vstack((last_segment, x[:2,:].T))
                last_segment = last_segment[-SGMNT_LEN:]
                if change["code"] == 0:
                    out.append(x[:2])
                    
                if DEBUG:
                    print "kf", x.T
                
        return out[:steps]    
        
if __name__ == "__main__":
    
    picklename = "inputs/trained_hexbug.p"
    
    if os.path.isfile(picklename):
        hb_predict = pickle.load(open(picklename, 'rb'))
    else:
        # 1. train
        filename = "inputs/training_data.txt"
        input_file = open(filename, 'r')
        
        # read data
        buff = input_file.readline()
        xy_coor = []
        while buff != "":
            xy_coor.append([int(i) for i in buff.split(",")])
            buff = input_file.readline()
        xy_coor = np.array(xy_coor, dtype=np.float_)
            
        # gaussian filter
        for i in xrange(xy_coor.shape[1]):
            xy_coor[:,i] = gaussian_filter(xy_coor[:,i], sigma=3)
        
        hb_predict = HexbugPredictor()
        hb_predict.learn_trajectory_segments(xy_coor)
        # hb_predict.learn_markov_model(xy_coor)
        
        pickle.dump(hb_predict, open(picklename, 'wb'), pickle.HIGHEST_PROTOCOL)
        
    filename = sys.argv[1]
    input_file = open(filename, 'r')
    
    buff = input_file.readline()
    xy_coor = []
    while buff != "":
        xy_coor.append([int(i) for i in buff.split(",")])
        buff = input_file.readline()
    xy_coor = np.array(xy_coor, dtype=np.float_)

    """
    # gaussian filter
    for i in xrange(xy_coor.shape[1]):
        xy_coor[:,i] = gaussian_filter(xy_coor[:,i], sigma=3)
    """

    if DEBUG:
        print filename
    
    last = -30 # the number of data to observe before making predictions
    out = hb_predict.predict(xy_coor[last:,:], 60)
    output_file = open('prediction.txt', 'w')
    for x,y in out:
        output_file.write("%d,%d\n" % (int(round(x)), int(round(y))))
    output_file.close()
