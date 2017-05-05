"""Problem Set 8: Motion History Images."""

import numpy as np
import cv2

import os

# I/O directories
input_dir = "input"
output_dir = "output"

DEBUG = False # debug variable

# Assignment code
class MotionHistoryBuilder(object):
    """Builds a motion history image (MHI) from sequential video frames."""
    # morphological OPEN kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    def __init__(self, frame, **kwargs):
        """Initialize motion history builder object.

        Parameters
        ----------
            frame: color BGR uint8 image of initial video frame, values in [0, 255]
            kwargs: additional keyword arguments needed by builder, e.g. theta, tau
        """
        # initialize variables
        self.mhi = np.zeros(frame.shape[:2], dtype=np.float_)  # e.g. motion history image
        self.I_tm1 = self.clean(frame) # image at t - 1
        
        # read keyword arguments
        self.b_theta = kwargs.get('b_theta', 3)  # extract binary image theta (default: 3)
        self.tau = kwargs.get('tau', 15)  # extract tau (default: 15)

    def clean(self, img):
        # convert to black and white, and to float data type
        img2 = np.round(0.12 * img[:,:,0] + 0.58 * img[:,:,1] + 0.3 * img[:,:,2]) 
        # img2 = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
        img2 = cv2.GaussianBlur(img2, (15,15), 0)
        img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, self.kernel)
        return img2
        
    def process(self, frame):
        """Process a frame of video, return binary image indicating motion areas.

        Parameters
        ----------
            frame: color BGR uint8 image of current video frame, values in [0, 255]

        Returns
        -------
            motion_image: binary image (type: bool or uint8), values: 0 (static) or 1 (moving)
        """
        I_t = self.clean(frame) # image at t
        
        # compute binary motion image
        threshold_index = np.abs(I_t-self.I_tm1)>=self.b_theta
        motion_image = np.zeros(frame.shape[:2], dtype=np.uint8)
        motion_image[threshold_index] = 1
        
        # update MHI
        self.mhi[self.mhi>0] -= 1
        self.mhi[threshold_index] = self.tau
        
        # update variables
        self.I_tm1 = I_t
        
        return motion_image  # note: make sure you return a binary image with 0s and 1s

    def get_MHI(self):
        """Return motion history image computed so far.

        Returns
        -------
            mhi: float motion history image, values in [0.0, 1.0]
        """
        # MHI is updated in process(). final steps are performed here
        # Note: This method may not be called for every frame (typically, only once)
        self.mhi /= self.tau # normalize to [0, 1]
        return self.mhi

class Moments(object):
    """Spatial moments of an image - unscaled and scaled."""

    def __init__(self, image):
        """Compute spatial moments on given image.

        Parameters
        ----------
            image: single-channel image, uint8 or float
        """
        # compute all desired moments here (recommended)
        x, y = np.meshgrid(np.arange(image.shape[1]), 
                           np.arange(image.shape[0]))
        M00 = (1.0 * (x**0) * (y**0) * image).sum()
        x_mean = (1.0 * (x**1) * (y**0) * image).sum() / M00
        y_mean = (1.0 * (x**0) * (y**1) * image).sum() / M00
        x2, y2 = 1.0 * x - x_mean, 1.0 * y - y_mean
        x2_pow = [x2**i for i in xrange(4)]
        y2_pow = [y2**i for i in xrange(4)]
        
        # array: [mu20, mu11, mu02, mu30, mu21, mu12, mu03, mu22]
        moments = np.array([(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3),(2,2)])
        self.central_moments = [(x2_pow[p]*y2_pow[q]*image).sum() for p,q in moments]
        
        u00 = ((x2**0) * (y2**0) * image).sum()
        # array: [nu20, nu11, nu02, nu30, nu21, nu12, nu03, nu22]
        self.scaled_moments = [self.central_moments[i] / (u00**(1.0+(moments[i,:].sum()/ 2.0))) for i in xrange(moments.shape[0])]
        # Note: Make sure computed moments are in correct order
        
        """
        test = cv2.moments(image)
        test1 = [test[i] for i in ['mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03']]
        test2 = [test[i] for i in ['nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']]
        test_x, test_y = test['m10'] / test['m00'], test['m01'] / test['m00']
        if abs(M00 - test['m00']) > 1e-10:
            print M00, test['m00']
        if abs(test_x - x_mean) > 1e-10:
            print x_mean, test_x
        if abs(test_y - y_mean) > 1e-10:
            print y_mean, test_y
        for i in xrange(len(test1)):
            if abs(test1[i]-self.central_moments[i]) > 1e-3:
                print 'c', i, self.central_moments[i], test1[i]
        for i in xrange(len(test2)):
            if abs(test2[i]-self.scaled_moments[i]) > 1e-10:
                print 's', i, self.scaled_moments[i], test2[i]
        """

    def get_central_moments(self):
        """Return central moments as NumPy array.

        Order: [mu20, mu11, mu02, mu30, mu21, mu12, mu03, mu22]

        Returns
        -------
            self.central_moments: float array of central moments
        """
        return self.central_moments

    def get_scaled_moments(self):
        """Return scaled central moments as NumPy array.

        Order: [nu20, nu11, nu02, nu30, nu21, nu12, nu03, nu22]

        Returns
        -------
            self.scaled_moments: float array of scaled central moments
        """
        return self.scaled_moments  # note: make sure moments are in correct order


def compute_feature_difference(a_features, b_features):
    """Compute feature difference between two videos.

    Parameters
    ----------
        a_features: feaures from one video, MHI & MEI moments in a 16-element 1D array
        b_features: like a_features, from other video

    Returns
    -------
        diff: a single float value, difference between the two feature vectors
    """
    # return feature difference using an appropriate measure
    # Tip: Scale/weight difference values to get better results as moment magnitudes differ
    diff = np.sqrt(np.power(a_features-b_features,2).sum())
    return diff


# Driver/helper code
def build_motion_history_image(builder_class, video_filename, save_frames={}, mhi_frame=None, mhi_filename=None, **kwargs):
    """Instantiate and run a motion history builder on a given video, return MHI.

    Creates an object of type builder_class, passing in initial video frame,
    and any additional keyword arguments.

    Parameters
    ----------
        builder_class: motion history builder class to instantiate
        video_filename: path to input video file
        save_frames: output binary motion images to save {<frame number>: <filename>}
        mhi_frame: which frame to obtain the motion history image at
        mhi_filename: output filename to save motion history image
        kwargs: arbitrary keyword arguments passed on to constructor

    Returns
    -------
        mhi: float motion history image generated by builder, values in [0.0, 1.0]
    """

    # Open video file
    video = cv2.VideoCapture(video_filename)
    print "Video: {} ({}x{}, {:.2f} fps, {} frames)".format(
        video_filename,
        int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)),
        video.get(cv2.cv.CV_CAP_PROP_FPS),
        int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)))

    # Initialize objects
    mhi_builder = None
    mhi = None
    frame_num = 0

    # Loop over video (till last frame or Ctrl+C is presssed)
    while True:
        try:
            # Try to read a frame
            okay, frame = video.read()
            if not okay:
                break  # no more frames, or can't read video

            # Initialize motion history builder (one-time only)
            if mhi_builder is None:
                mhi_builder = builder_class(frame, **kwargs)

            # Process frame
            motion_image = mhi_builder.process(frame)

            # Save output, if indicated
            if frame_num in save_frames:
                cv2.imwrite(save_frames[frame_num], np.uint8(motion_image * 255))  # scale [0, 1] => [0, 255]
            
            # Grab MHI, if indicated
            if frame_num == mhi_frame:
                mhi = mhi_builder.get_MHI()
                print "MHI frame: {}".format(mhi_frame)
                break  # uncomment for early stop

            # Update frame number
            frame_num += 1
        except KeyboardInterrupt:  # press ^C to quit
            break

    # If not obtained earlier, get MHI now
    if mhi is None:
        mhi = mhi_builder.get_MHI()

    # Save MHI, if filename is given
    if mhi_filename is not None:
        cv2.imwrite(mhi_filename, np.uint8(mhi * 255))  # scale [0, 1] => [0, 255]

    return mhi


def match_features(a_features_dict, b_features_dict, n_actions):
    """Compare features, tally matches for each action pair to produce a confusion matrix.

    Note: Skips comparison for keys that are identical in the two dicts.

    Parameters
    ----------
        a_features_dict: one set of features, as a dict with key: (<action>, <participant>, <trial>)
        b_features_dict: another set of features like a_features
        n_actions: number of distinct actions present in the feature sets

    Returns
    -------
        confusion_matrix: table of matches found, n_actions by n_actions
    """
    
    confusion_matrix = np.zeros((n_actions, n_actions), dtype=np.float_)
    for a_key, a_features in a_features_dict.iteritems():
        min_diff = np.inf
        best_match = None
        for b_key, b_features in b_features_dict.iteritems():
            if a_key == b_key:
                continue  # don't compare with yourself!
            diff = compute_feature_difference(a_features, b_features)
            if diff < min_diff:
                min_diff = diff
                best_match = b_key
        if best_match is not None:
            if DEBUG:
                print "{} matches {}, diff: {}".format(a_key, best_match, min_diff)  # [debug]
            confusion_matrix[a_key[0] - 1, best_match[0] - 1] += 1  # note: 1-based to 0-based indexing

    confusion_matrix /= confusion_matrix.sum(axis=1)[:, np.newaxis] # normalize confusion_matrix along each row
    return confusion_matrix


def main():
    # 1a
    build_motion_history_image(MotionHistoryBuilder,  # motion history builder class
        os.path.join(input_dir, "PS8A1P1T1.avi"),  # input video
        save_frames={
            10: os.path.join(output_dir, 'ps8-1-a-1.png'),
            20: os.path.join(output_dir, 'ps8-1-a-2.png'),
            30: os.path.join(output_dir, 'ps8-1-a-3.png')
        }, # output motion images to save, mapped to filenames
        b_theta=3, tau=15 # Specify any other keyword args that your motion history builder expects, e.g. theta, tau
        ) 
    # 1b
    test_cases = [{'in': 'PS8A1P1T1.avi', 'out': 'ps8-1-b-1.png',
                   'mhi_frame': 109, 'tau': 109}, # 0->66->109
                  {'in': 'PS8A2P1T1.avi', 'out': 'ps8-1-b-2.png',
                   'mhi_frame': 55, 'tau': 25}, # 31->37->42->48->55
                  {'in': 'PS8A3P1T3.avi', 'out': 'ps8-1-b-3.png',
                   'mhi_frame': 97, 'tau': 97}] # 0->55->97
    for test in test_cases:
        build_motion_history_image(MotionHistoryBuilder,  # motion history builder class
            os.path.join(input_dir, test['in']),  # choose sequence (person, trial) for action A1
            mhi_frame=test['mhi_frame'],  # pick a good frame to obtain MHI at, i.e. when action just ends
            mhi_filename=os.path.join(output_dir, test['out']),
            b_theta=3, tau=test['tau'] # Specify any other keyword args that your motion history builder expects, e.g. theta, tau
            )
        
    # 2a
    # Compute MHI and MEI features (unscaled and scaled central moments) for each video
    central_moment_features = {}  # 16 features (8 MHI, 8 MEI) as one vector, key: (<action>, <participant>, <trial>)
    scaled_moment_features = {}  # similarly, scaled central moments
    
    default_params = dict(mhi_frame=110, tau=110, b_theta=25)  # params for build_motion_history(), overriden by custom_params for specified videos
    # Note: To specify custom parameters for a video, add to the dict below:
    #   (<action>, <participant>, <trial>): dict(<param1>=<value1>, <param2>=<value2>, ...)
    custom_params = {
        (2, 1, 1): dict(mhi_frame=55, tau=25),
        (2, 1, 2): dict(mhi_frame=55, tau=25), 
        (2, 1, 3): dict(mhi_frame=55, tau=25), 
        (2, 2, 1): dict(mhi_frame=50, tau=30), 
        (2, 2, 2): dict(mhi_frame=50, tau=30), 
        (2, 2, 3): dict(mhi_frame=50, tau=30), 
        (2, 3, 1): dict(mhi_frame=51, tau=23), 
        (2, 3, 2): dict(mhi_frame=51, tau=23), 
        (2, 3, 3): dict(mhi_frame=51, tau=23)
    }

    # Loop for each action, participant, trial
    n_actions = 3
    n_participants = 3
    n_trials = 3
    
    print "Computing features for each video..."
    for a in xrange(1, n_actions + 1):  # actions
        for p in xrange(1, n_participants + 1):  # participants
            for t in xrange(1, n_trials + 1):  # trials
                video_filename = os.path.join(input_dir, "PS8A{}P{}T{}.avi".format(a, p, t))
                if DEBUG:
                    mhi = build_motion_history_image(MotionHistoryBuilder, video_filename,
                                                     mhi_filename=os.path.join(output_dir, "A{}P{}T{}.png".format(a, p, t)),
                                                     **dict(default_params, **custom_params.get((a, p, t), {})))
                else:
                    mhi = build_motion_history_image(MotionHistoryBuilder, video_filename,
                                                     **dict(default_params, **custom_params.get((a, p, t), {})))
                #cv2.imshow("MHI: PS8A{}P{}T{}.avi".format(a, p, t), mhi)  # [debug]
                #cv2.waitKey(1)  # uncomment if using imshow
                mei = np.uint8(mhi > 0)
                mhi_moments = Moments(mhi)
                mei_moments = Moments(mei)
                central_moment_features[(a, p, t)] = np.hstack((mhi_moments.get_central_moments(), mei_moments.get_central_moments()))
                scaled_moment_features[(a, p, t)] = np.hstack((mhi_moments.get_scaled_moments(), mei_moments.get_scaled_moments()))
    
    # Match features in a leave-one-out scheme (each video with all others)
    central_moments_confusion = match_features(central_moment_features, central_moment_features, n_actions)
    print "Confusion matrix (unscaled central moments):-"
    print central_moments_confusion

    # Similarly with scaled moments
    scaled_moments_confusion = match_features(scaled_moment_features, scaled_moment_features, n_actions)
    print "Confusion matrix (scaled central moments):-"
    print scaled_moments_confusion
    
    # 2b
    # Match features by testing one participant at a time (i.e. taking them out)
    # Note: Pick one between central_moment_features and scaled_moment_features
    confusion_P = {}
    for p in [1, 2, 3]:
        features_P = {key: feature for key, feature in scaled_moment_features.iteritems() if key[1] == p}
        features_sans_P = {key: feature for key, feature in scaled_moment_features.iteritems() if key[1] != p}
        confusion_P[p] = match_features(features_P, features_sans_P, n_actions)
        print "Confusion matrix for P{}:-".format(p)
        print confusion_P[p]
    print "Confusion matrix for all P:-"
    print (confusion_P[1] + confusion_P[2] + confusion_P[3]) / 3.0
    # Note: Feel free to modify this driver function, but do not modify the interface for other functions/methods!
    
if __name__ == "__main__":
    main()
