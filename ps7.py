"""Problem Set 7: Particle Filter Tracking."""

import numpy as np
import cv2

import os

# I/O directories
input_dir = "input"
output_dir = "output"

DEBUG = False # debug mode
SHOW = False # show movie window

# Assignment code
class ParticleFilter(object):
    """A particle filter tracker, encapsulating state, initialization and update methods."""

    def __init__(self, frame, template, **kwargs):
        """Initialize particle filter object.

        Parameters
        ----------
            frame: color BGR uint8 image of initial video frame, values in [0, 255]
            template: color BGR uint8 image of patch to track, values in [0, 255]
            kwargs: keyword arguments needed by particle filter model, including:
            - num_particles: number of particles
            - init_method: random (choose random dots in image), template_match (randomly select points where MSE is low)
        """
        self.num_particles = kwargs.get('num_particles', 100)  # extract num_particles (default: 100)
        self.init_method = kwargs.get('init_method', 'random')  # extract initial method (default: random)
        self.sigma_MSE = kwargs.get('sigma_MSE', 10) # extract sigma_MSE (default: 10)
        self.dynamics_error = kwargs.get('dynamics_error', 10) # extract dynamics_error (default: 10)
        
        # Your code here - extract any additional keyword arguments you need and initialize state
        self.template = np.round(0.12 * template[:,:,0] + 0.58 * template[:,:,1] + 0.3 * template[:,:,2]).astype(np.uint8)
        frame = np.round(0.12 * frame[:,:,0] + 0.58 * frame[:,:,1] + 0.3 * frame[:,:,2]).astype(np.uint8)
        self.scale = 1.0
        
        # initial distribution of particles
        yLen1, xLen1 = self.template.shape[:2]
        yLen2, xLen2 = frame.shape[0]-yLen1+yLen1/2+1, frame.shape[1]-xLen1+xLen1/2+1
        yLen1, xLen1 = yLen1/2, xLen1/2
                 
        if self.init_method == 'template_match':
            uInit, vInit = np.meshgrid(xrange(xLen1,xLen2), xrange(yLen1,yLen2))
            wInit = cv2.matchTemplate(frame, self.template, cv2.TM_CCORR_NORMED)
            wInit += np.random.normal(0,0.01,wInit.shape)
            self.x = np.concatenate((uInit.reshape(-1,1),vInit.reshape(-1,1),wInit.reshape(-1,1)), axis=1)
            self.x = self.x[self.x[:,2].argsort()[-1:-1-self.num_particles:-1]]
        elif self.init_method == 'mix_match_and_random':
            n_match = self.num_particles / 2
            uInit, vInit = np.meshgrid(xrange(xLen1,xLen2), xrange(yLen1,yLen2))
            wInit = cv2.matchTemplate(frame, self.template, cv2.TM_CCORR_NORMED)
            wInit += np.random.normal(0,0.01,wInit.shape)
            x_match = np.concatenate((uInit.reshape(-1,1),vInit.reshape(-1,1),wInit.reshape(-1,1)), axis=1)
            x_match = x_match[x_match[:,2].argsort()[-1:-1-n_match:-1]]

            n_rand = self.num_particles - n_match
            uInit = np.random.randint(xLen1, xLen2, size=(n_rand,1))
            vInit = np.random.randint(yLen1, yLen2, size=(n_rand,1))
            wInit = np.ones((n_rand,1), dtype=np.float_)
            x_rand = np.concatenate((uInit,vInit,wInit), axis=1)
            
            self.x = np.concatenate((x_match, x_rand), axis=0)
        else: # random
            uInit = np.random.randint(xLen1, xLen2, size=(self.num_particles,1))
            vInit = np.random.randint(yLen1, yLen2, size=(self.num_particles,1))
            wInit = np.ones((self.num_particles,1), dtype=np.float_)
            self.x = np.concatenate((uInit,vInit,wInit), axis=1)
        self.x[:,2] /= self.x[:,2].sum() # normalize weights
        
    def process(self, frame):
        """Process a frame (image) of video and update filter state.

        Parameters
        ----------
            frame: color BGR uint8 image of current video frame, values in [0, 255]
        """
        # Your code here - use the frame as a new observation (measurement) and update model
        frame = 0.12 * frame[:,:,0] + 0.58 * frame[:,:,1] + 0.3 * frame[:,:,2]
        # sample from discrete distribution given w(t-1)
        d_t = np.random.multinomial(self.num_particles, self.x[:,2])
        # sample x(t) from p(xt | xt-1, ut)
        u_t = np.repeat(self.x[:,0],d_t).reshape(-1,1)
        v_t = np.repeat(self.x[:,1],d_t).reshape(-1,1)
        u_t = u_t + np.random.normal(0, self.dynamics_error, u_t.shape)
        v_t = v_t + np.random.normal(0, self.dynamics_error, v_t.shape)
        # handle noise that cause the particle to go out of bounds
        tyLen, txLen = self.template.shape[:2]
        u_t = np.clip(u_t, txLen/2, frame.shape[1]-txLen+txLen/2)
        v_t = np.clip(v_t, tyLen/2, frame.shape[0]-tyLen+tyLen/2)
        # reweight
        u_t2 = u_t - txLen/2
        v_t2 = v_t - tyLen/2
        w_t2 = np.ones(u_t.shape, dtype=np.float_) 
        for i in xrange(w_t2.size):
            u1, u2, v1, v2 = u_t2[i], u_t2[i]+txLen, v_t2[i], v_t2[i]+tyLen
            mse = (self.template - frame[v1:v2,u1:u2])**2
            w_t2[i] = 1.0 * mse.sum() / mse.size
        w_t = np.exp(-w_t2/(2.0*(self.sigma_MSE**2)))
        # update normalization factor
        self.x = np.concatenate((u_t,v_t,w_t), axis=1)
        # normalize weights
        self.x[:,2] /= self.x[:,2].sum()
        self.u_mean = (self.x[:,0] * self.x[:,2]).sum()
        self.v_mean = (self.x[:,1] * self.x[:,2]).sum()

    def render(self, frame_out):
        """Visualize current particle filter state.

        Parameters
        ----------
            frame_out: copy of frame to overlay visualization on
        """
        # Note: This may not be called for all frames, so don't do any model updates here!
        # draw particles
        for u, v in self.x[:,0:2]:
            cv2.circle(frame_out, (int(u),int(v)), 0, (0,0,255), thickness=3) 
        
        # tracking window
        yLen1, xLen1 = self.template.shape[:2]
        if self.scale != 1.0:
            yLen1, xLen1 = self.scale * yLen1, self.scale * xLen1
        u1, v1 = int(round(self.u_mean-xLen1/2)), int(round(self.v_mean-yLen1/2))
        u2, v2 = int(round(u1+xLen1)), int(round(v1+yLen1))
        cv2.rectangle(frame_out,(u1,v1),(u2,v2),(255,0,0),2)
        
        # a circle to indicate spread
        d = np.sqrt((self.x[:,0]-self.u_mean)**2 + (self.x[:,1]-self.v_mean)**2)
        r = (d * self.x[:,2]).sum()
        cv2.circle(frame_out, (int(round(self.u_mean)),int(round(self.v_mean))), int(round(r)), (0,255,0), thickness=2)


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker that updates its appearance model over time."""

    def __init__(self, frame, template, **kwargs):
        """Initialize appearance model particle filter object (parameters same as ParticleFilter)."""
        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor
        # Your code here - additional initialization steps, keyword arguments
        self.template_alpha = kwargs.get('template_alpha', 0.5) # extract template_alpha (default: 0.5)


    def process(self, frame):
        """Process a frame (image) of video and update filter state. (parameters same as ParticleFilter)."""
        super(AppearanceModelPF, self).process(frame)  # call base process
        # Override process() to implement appearance model update
        yLen1, xLen1 = self.template.shape[:2]
        u1, v1 = int(round(self.u_mean-xLen1/2)), int(round(self.v_mean-yLen1/2))
        u2, v2 = u1+xLen1, v1+yLen1
        best = frame[v1:v2,u1:u2]
        best = 0.12 * best[:,:,0] + 0.58 * best[:,:,1] + 0.30 * best[:,:,2]
        self.template = np.round(self.template_alpha * best + (1.0 - self.template_alpha) * self.template).astype(np.uint8)

    # Override render() if desired (shouldn't have to, ideally)

class MeanShiftModelPF(ParticleFilter):
    """A variation of particle filter tracker that updates its appearance model over time."""

    def __init__(self, frame, template, **kwargs):
        """Initialize appearance model particle filter object (parameters same as ParticleFilter)."""
        super(MeanShiftModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor
        # Your code here - additional initialization steps, keyword arguments
        self.hist_bin = kwargs.get('hist_bin', 8) # extract hist_bin (default: 8)
        self.template = template
        self.template_hist = self.get_hist(template)
        
    def get_hist(self, image):
        """Process image and return histogram representation

        Parameters
        ----------
            image: color BGR uint8 image, values in [0, 255]
        """
        b, bin_edges = np.histogram(image[:,:,0], bins=self.hist_bin, range=(0,255))
        g, bin_edges = np.histogram(image[:,:,1], bins=self.hist_bin, range=(0,255))
        r, bin_edges = np.histogram(image[:,:,2], bins=self.hist_bin, range=(0,255))
        h = np.concatenate((b,g,r), axis=0)
        return h.astype(np.float_) / h.sum()

    def process(self, frame):
        """Process a frame (image) of video and update filter state.

        Parameters
        ----------
            frame: color BGR uint8 image of current video frame, values in [0, 255]
        """
        # Your code here - use the frame as a new observation (measurement) and update model
        # sample from discrete distribution given w(t-1)
        d_t = np.random.multinomial(self.num_particles, self.x[:,2])
        # sample x(t) from p(xt | xt-1, ut)
        u_t = np.repeat(self.x[:,0],d_t).reshape(-1,1)
        v_t = np.repeat(self.x[:,1],d_t).reshape(-1,1)
        u_t = u_t + np.random.normal(0, self.dynamics_error, u_t.shape)
        v_t = v_t + np.random.normal(0, self.dynamics_error, v_t.shape)
        # handle noise that cause the particle to go out of bounds
        tyLen, txLen = self.template.shape[:2]
        u_t = np.clip(u_t, txLen/2, frame.shape[1]-txLen+txLen/2)
        v_t = np.clip(v_t, tyLen/2, frame.shape[0]-tyLen+tyLen/2)
        # reweight
        u_t2 = u_t - txLen/2
        v_t2 = v_t - tyLen/2
        w_t2 = np.ones(u_t.shape, dtype=np.float_) 
        for i in xrange(w_t2.size):
            u1, u2, v1, v2 = u_t2[i], u_t2[i]+txLen, v_t2[i], v_t2[i]+tyLen
            frame_hist = self.get_hist(frame[v1:v2,u1:u2])
            csx2 = (self.template_hist - frame_hist)**2 / (self.template_hist + frame_hist)
            w_t2[i] = 0.5 * np.nansum(csx2)
        w_t = np.exp(-w_t2/(2.0*(self.sigma_MSE**2)))
        # update normalization factor
        self.x = np.concatenate((u_t,v_t,w_t), axis=1)
        # normalize weights
        self.x[:,2] /= self.x[:,2].sum()
        self.u_mean = (self.x[:,0] * self.x[:,2]).sum()
        self.v_mean = (self.x[:,1] * self.x[:,2]).sum()

    # Override render() if desired (shouldn't have to, ideally)

class MoreDynamicsModelPF(ParticleFilter):
    """A variation of particle filter tracker that updates its appearance model over time."""

    def __init__(self, frame, template, **kwargs):
        """Initialize appearance model particle filter object (parameters same as ParticleFilter)."""
        super(MoreDynamicsModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor
        # Your code here - additional initialization steps, keyword arguments
        self.scale_error = kwargs.get('scale_error', 0.001) # scale (default: 0.001)
        self.scale_min = kwargs.get('scale_min', 0.25) # scale (default: 0.25)
        sInit = np.ones((self.num_particles,1), dtype=np.float_)
        self.x = np.concatenate((self.x,sInit), axis=1)
        self.best_mse = 0

    def process(self, frame):
        """Process a frame (image) of video and update filter state.

        Parameters
        ----------
            frame: color BGR uint8 image of current video frame, values in [0, 255]
        """
        # Your code here - use the frame as a new observation (measurement) and update model
        frame = 0.12 * frame[:,:,0] + 0.58 * frame[:,:,1] + 0.3 * frame[:,:,2]
        # sample from discrete distribution given w(t-1)
        d_t = np.random.multinomial(self.num_particles, self.x[:,2])
        # sample x(t) from p(xt | xt-1, ut)
        u_t = np.repeat(self.x[:,0],d_t).reshape(-1,1)
        v_t = np.repeat(self.x[:,1],d_t).reshape(-1,1)
        s_t = np.repeat(self.x[:,3],d_t).reshape(-1,1)
        
        u_t = u_t + np.random.normal(0, self.dynamics_error, u_t.shape)
        v_t = v_t + np.random.normal(0, self.dynamics_error, v_t.shape)
        s_t = s_t - np.abs(np.random.normal(0, self.scale_error, s_t.shape))
        # handle noise that cause the particle to go out of bounds
        tyLen = np.round(s_t * self.template.shape[0]).astype(np.int_)
        txLen = np.round(s_t * self.template.shape[1]).astype(np.int_)
        u_t = np.clip(u_t, txLen/2, frame.shape[1]-txLen+txLen/2)
        v_t = np.clip(v_t, tyLen/2, frame.shape[0]-tyLen+tyLen/2)
        s_t = np.clip(s_t, self.scale_min, 1.0)
        # reweight
        u_t2 = u_t - txLen/2
        v_t2 = v_t - tyLen/2
        w_t2 = np.ones(u_t.shape, dtype=np.float_) 
        for i in xrange(w_t2.size):
            u1, u2, v1, v2 = u_t2[i], u_t2[i]+txLen[i], v_t2[i], v_t2[i]+tyLen[i]
            frame2 = cv2.resize(frame[v1:v2,u1:u2], self.template.shape[::-1])
            mse = (self.template - frame2)**2
            w_t2[i] = 1.0 * mse.sum() / mse.size
        w_t = np.exp(-w_t2/(2.0*(self.sigma_MSE**2)))
        # update normalization factor
        self.x = np.concatenate((u_t,v_t,w_t,s_t), axis=1)
        # normalize weights
        self.x[:,2] /= self.x[:,2].sum()
        self.u_mean = (self.x[:,0] * self.x[:,2]).sum()
        self.v_mean = (self.x[:,1] * self.x[:,2]).sum()
        self.scale = (self.x[:,3] * self.x[:,2]).sum()
        
# Driver/helper code
def get_template_rect(rect_filename):
    """Read rectangular template bounds from given file.

    The file must define 4 numbers (floating-point or integer), separated by whitespace:
    <x> <y>
    <w> <h>

    Parameters
    ----------
        rect_filename: path to file defining template rectangle

    Returns
    -------
        template_rect: dictionary specifying template bounds (x, y, w, h), as float or int

    """
    with open(rect_filename, 'r') as f:
        values = [float(v) for v in f.read().split()]
        return dict(zip(['x', 'y', 'w', 'h'], values[0:4]))


def run_particle_filter(pf_class, video_filename, template_rect, save_frames={}, **kwargs):
    """Instantiate and run a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any keyword arguments.

    Parameters
    ----------
        pf_class: particle filter class to instantiate (e.g. ParticleFilter)
        video_filename: path to input video file
        template_rect: dictionary specifying template bounds (x, y, w, h), as float or int
        save_frames: dictionary of frames to save {<frame number>|'template': <filename>}
        kwargs: arbitrary keyword arguments passed on to particle filter class
    """

    # Open video file
    video = cv2.VideoCapture(video_filename)

    # Initialize objects
    template = None
    pf = None
    frame_num = 0

    # Loop over video (till last frame or Ctrl+C is presssed)
    while True:
        try:
            # Try to read a frame
            okay, frame = video.read()
            if not okay:
                break  # no more frames, or can't read video

            # Extract template and initialize (one-time only)
            if template is None:
                template = frame[int(template_rect['y']):int(template_rect['y'] + template_rect['h']),
                                 int(template_rect['x']):int(template_rect['x'] + template_rect['w'])]
                if 'template' in save_frames:
                    cv2.imwrite(save_frames['template'], template)
                pf = pf_class(frame, template, **kwargs)

            # Process frame
            pf.process(frame)

            # Render and save output, if indicated
            if frame_num in save_frames:
                frame_out = frame.copy()
                pf.render(frame_out)
                cv2.imwrite(save_frames[frame_num], frame_out)

            if SHOW:
                if frame_num not in save_frames:
                    frame_out = frame.copy()
                    pf.render(frame_out)
                cv2.imshow('frame', frame_out)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Update frame number
            frame_num += 1
        except KeyboardInterrupt:  # press ^C to quit
            break


def main():
    # Note: Comment out parts of this code as necessary
    # init_method = 'random'
    init_method = 'mix_match_and_random'
    # 1a
    run_particle_filter(ParticleFilter,  # particle filter model class
        os.path.join(input_dir, "pres_debate.avi"),  # input video
        get_template_rect(os.path.join(input_dir, "pres_debate.txt")),  # suggested template window (dict)
        {
            'template': os.path.join(output_dir, 'ps7-1-a-1.png'),
            0: os.path.join(output_dir, 'ps7-1-a-1a.png'),
            28: os.path.join(output_dir, 'ps7-1-a-2.png'),
            84: os.path.join(output_dir, 'ps7-1-a-3.png'),
            144: os.path.join(output_dir, 'ps7-1-a-4.png')
        } if DEBUG else {
            'template': os.path.join(output_dir, 'ps7-1-a-1.png'),
            28: os.path.join(output_dir, 'ps7-1-a-2.png'),
            84: os.path.join(output_dir, 'ps7-1-a-3.png'),
            144: os.path.join(output_dir, 'ps7-1-a-4.png')
        },  # frames to save, mapped to filenames, and 'template' if desired
        num_particles=50, sigma_MSE=10, dynamics_error=10, init_method=init_method)
        
    if DEBUG:
        # 1b
        # Repeat 1a, but vary template window size and discuss trade-offs (no output images required)
        run_particle_filter(ParticleFilter,  # particle filter model class
            os.path.join(input_dir, "pres_debate.avi"),  # input video
            {'y': 239.7-40.0, 'x': 372.65-30.0, 'w': 2*25.0, 'h': 2*33.0},  # suggested template window (dict)
            {
                'template': os.path.join(output_dir, 'ps7-1-b-1.png'),
                0: os.path.join(output_dir, 'ps7-1-b-1a.png'),
                28: os.path.join(output_dir, 'ps7-1-b-1b.png'),
                84: os.path.join(output_dir, 'ps7-1-b-1c.png'),
                144: os.path.join(output_dir, 'ps7-1-b-1d.png')
            },  # frames to save, mapped to filenames, and 'template' if desired
            num_particles=50, sigma_MSE=10, dynamics_error=10, init_method=init_method)
        
        run_particle_filter(ParticleFilter,  # particle filter model class
            os.path.join(input_dir, "pres_debate.avi"),  # input video
            {'y': 239.7-130.0, 'x': 372.65-103.0, 'w': 2*103.0, 'h': 2*130.0},  # suggested template window (dict)
            {
                'template': os.path.join(output_dir, 'ps7-1-b-2.png'),
                0: os.path.join(output_dir, 'ps7-1-b-2a.png'),
                28: os.path.join(output_dir, 'ps7-1-b-2b.png'),
                84: os.path.join(output_dir, 'ps7-1-b-2c.png'),
                144: os.path.join(output_dir, 'ps7-1-b-2d.png')
            },  # frames to save, mapped to filenames, and 'template' if desired
            num_particles=50, sigma_MSE=10, dynamics_error=10, init_method=init_method)
    
        # 1c
        # Repeat 1a, but vary the sigma_MSE parameter (no output images required)
        run_particle_filter(ParticleFilter,  # particle filter model class
            os.path.join(input_dir, "pres_debate.avi"),  # input video
            get_template_rect(os.path.join(input_dir, "pres_debate.txt")),  # suggested template window (dict)
            {
                'template': os.path.join(output_dir, 'ps7-1-c-1.png'),
                0: os.path.join(output_dir, 'ps7-1-c-1a.png'),
                28: os.path.join(output_dir, 'ps7-1-c-1b.png'),
                84: os.path.join(output_dir, 'ps7-1-c-1c.png'),
                144: os.path.join(output_dir, 'ps7-1-c-1d.png')
            },  # frames to save, mapped to filenames, and 'template' if desired
            num_particles=50, sigma_MSE=3, dynamics_error=10, init_method=init_method)
        run_particle_filter(ParticleFilter,  # particle filter model class
            os.path.join(input_dir, "pres_debate.avi"),  # input video
            get_template_rect(os.path.join(input_dir, "pres_debate.txt")),  # suggested template window (dict)
            {
                'template': os.path.join(output_dir, 'ps7-1-c-2.png'),
                0: os.path.join(output_dir, 'ps7-1-c-2a.png'),
                28: os.path.join(output_dir, 'ps7-1-c-2b.png'),
                84: os.path.join(output_dir, 'ps7-1-c-2c.png'),
                144: os.path.join(output_dir, 'ps7-1-c-2d.png')
            },  # frames to save, mapped to filenames, and 'template' if desired
            num_particles=50, sigma_MSE=100, dynamics_error=10, init_method=init_method)

        # 1d
        # Repeat 1a, but try to optimize (minimize) num_particles (no output images required)
        run_particle_filter(ParticleFilter,  # particle filter model class
            os.path.join(input_dir, "pres_debate.avi"),  # input video
            get_template_rect(os.path.join(input_dir, "pres_debate.txt")),  # suggested template window (dict)
            {
                'template': os.path.join(output_dir, 'ps7-1-d-1.png'),
                0: os.path.join(output_dir, 'ps7-1-d-1a.png'),
                28: os.path.join(output_dir, 'ps7-1-d-1b.png'),
                84: os.path.join(output_dir, 'ps7-1-d-1c.png'),
                144: os.path.join(output_dir, 'ps7-1-d-1d.png')
            },  # frames to save, mapped to filenames, and 'template' if desired
            num_particles=5, sigma_MSE=10, dynamics_error=10, init_method=init_method)
        run_particle_filter(ParticleFilter,  # particle filter model class
            os.path.join(input_dir, "pres_debate.avi"),  # input video
            get_template_rect(os.path.join(input_dir, "pres_debate.txt")),  # suggested template window (dict)
            {
                'template': os.path.join(output_dir, 'ps7-1-d-2.png'),
                0: os.path.join(output_dir, 'ps7-1-d-2a.png'),
                28: os.path.join(output_dir, 'ps7-1-d-2b.png'),
                84: os.path.join(output_dir, 'ps7-1-d-2c.png'),
                144: os.path.join(output_dir, 'ps7-1-d-2d.png')
            },  # frames to save, mapped to filenames, and 'template' if desired
            num_particles=200, sigma_MSE=10, dynamics_error=10, init_method=init_method)

    # 1e
    run_particle_filter(ParticleFilter,
        os.path.join(input_dir, "noisy_debate.avi"),
        get_template_rect(os.path.join(input_dir, "noisy_debate.txt")),
        {
            14: os.path.join(output_dir, 'ps7-1-e-1.png'),
            32: os.path.join(output_dir, 'ps7-1-e-2.png'),
            46: os.path.join(output_dir, 'ps7-1-e-3.png')
        },
        num_particles=50, sigma_MSE=10, dynamics_error=10, init_method=init_method)
        # Tune parameters so that model can continuing tracking through noise
    
    # 2a
    # Implement AppearanceModelPF (derived from ParticleFilter)
    # Run it on pres_debate.avi to track Romney's left hand, tweak parameters to track up to frame 140
    run_particle_filter(AppearanceModelPF,  # particle filter model class
        os.path.join(input_dir, "pres_debate.avi"),  # input video
        {'y': 385.0, 'x': 525.0, 'w': 90.0, 'h': 100.0},
        {
            'template': os.path.join(output_dir, 'ps7-2-a-1.png'),
            0: os.path.join(output_dir, 'ps7-2-a-1a.png'),
            15: os.path.join(output_dir, 'ps7-2-a-2.png'),
            50: os.path.join(output_dir, 'ps7-2-a-3.png'),
            140: os.path.join(output_dir, 'ps7-2-a-4.png')
        } if DEBUG else {
            'template': os.path.join(output_dir, 'ps7-2-a-1.png'),
            15: os.path.join(output_dir, 'ps7-2-a-2.png'),
            50: os.path.join(output_dir, 'ps7-2-a-3.png'),
            140: os.path.join(output_dir, 'ps7-2-a-4.png')
        },  # frames to save, mapped to filenames, and 'template' if desired
        num_particles=100, sigma_MSE=5, dynamics_error=10, 
        template_alpha=0.25, init_method=init_method)
        
    # 2b
    # Run AppearanceModelPF on noisy_debate.avi, tweak parameters to track hand up to frame 140
    run_particle_filter(AppearanceModelPF,  # particle filter model class
        os.path.join(input_dir, "noisy_debate.avi"),  # input video
        {'y': 385.0, 'x': 525.0, 'w': 90.0, 'h': 100.0},
        {
            'template': os.path.join(output_dir, 'ps7-2-b-1.png'),
            0: os.path.join(output_dir, 'ps7-2-b-1a.png'),
            15: os.path.join(output_dir, 'ps7-2-b-2.png'),
            50: os.path.join(output_dir, 'ps7-2-b-3.png'),
            140: os.path.join(output_dir, 'ps7-2-b-4.png')
        } if DEBUG else {
            'template': os.path.join(output_dir, 'ps7-2-b-1.png'),
            15: os.path.join(output_dir, 'ps7-2-b-2.png'),
            50: os.path.join(output_dir, 'ps7-2-b-3.png'),
            140: os.path.join(output_dir, 'ps7-2-b-4.png')
        },  # frames to save, mapped to filenames, and 'template' if desired
        num_particles=150, sigma_MSE=5, dynamics_error=10, 
        template_alpha=0.25, init_method=init_method)

    # EXTRA CREDIT
    # 3: Use color histogram distance instead of MSE (you can implement a derived class similar to AppearanceModelPF)
    run_particle_filter(MeanShiftModelPF,  # particle filter model class
        os.path.join(input_dir, "pres_debate.avi"),  # input video
        get_template_rect(os.path.join(input_dir, "pres_debate.txt")),  # suggested template window (dict)
        {
            'template': os.path.join(output_dir, 'ps7-3-a-1.png'),
            0: os.path.join(output_dir, 'ps7-3-a-1a.png'),
            28: os.path.join(output_dir, 'ps7-3-a-2.png'),
            84: os.path.join(output_dir, 'ps7-3-a-3.png'),
            144: os.path.join(output_dir, 'ps7-3-a-4.png')
        } if DEBUG else {
            'template': os.path.join(output_dir, 'ps7-3-a-1.png'),
            28: os.path.join(output_dir, 'ps7-3-a-2.png'),
            84: os.path.join(output_dir, 'ps7-3-a-3.png'),
            144: os.path.join(output_dir, 'ps7-3-a-4.png')
        },  # frames to save, mapped to filenames, and 'template' if desired
        num_particles=30, sigma_MSE=0.2, dynamics_error=10, hist_bin=8, init_method=init_method)
    
    run_particle_filter(MeanShiftModelPF,  # particle filter model class
        os.path.join(input_dir, "pres_debate.avi"),  # input video
        {'y': 385.0, 'x': 550.0, 'w': 40.0, 'h': 100.0},  # suggested template window (dict)
        {
            'template': os.path.join(output_dir, 'ps7-3-b-1.png'),
            0: os.path.join(output_dir, 'ps7-3-b-1a.png'),
            15: os.path.join(output_dir, 'ps7-3-b-2.png'),
            50: os.path.join(output_dir, 'ps7-3-b-3.png'),
            140: os.path.join(output_dir, 'ps7-3-b-4.png')
        } if DEBUG else {
            'template': os.path.join(output_dir, 'ps7-3-b-1.png'),
            15: os.path.join(output_dir, 'ps7-3-b-2.png'),
            50: os.path.join(output_dir, 'ps7-3-b-3.png'),
            140: os.path.join(output_dir, 'ps7-3-b-4.png')
        },  # frames to save, mapped to filenames, and 'template' if desired
        num_particles=100, sigma_MSE=0.1, dynamics_error=40, hist_bin=8, init_method=init_method)
    
    # 4: Implement a more sophisticated model to deal with occlusions and size/perspective changes
    run_particle_filter(MoreDynamicsModelPF,  # particle filter model class
        os.path.join(input_dir, "pedestrians.avi"),  # input video
        get_template_rect(os.path.join(input_dir, "pedestrians.txt")),  # suggested template window (dict)
        {
            'template': os.path.join(output_dir, 'ps7-4-a-1.png'),
            0: os.path.join(output_dir, 'ps7-4-a-1a.png'),
            40: os.path.join(output_dir, 'ps7-4-a-2.png'),
            100: os.path.join(output_dir, 'ps7-4-a-3.png'),
            133: os.path.join(output_dir, 'ps7-4-a-3a.png'),
            240: os.path.join(output_dir, 'ps7-4-a-4.png')
        } if DEBUG else {
            'template': os.path.join(output_dir, 'ps7-4-a-1.png'),
            40: os.path.join(output_dir, 'ps7-4-a-2.png'),
            100: os.path.join(output_dir, 'ps7-4-a-3.png'),
            240: os.path.join(output_dir, 'ps7-4-a-4.png')
        },  # frames to save, mapped to filenames, and 'template' if desired
        num_particles=100, sigma_MSE=30, dynamics_error=2, 
        ist_bin=8, scale_error=0.005, scale_min=0.25, init_method=init_method)
    
    if DEBUG:
        run_particle_filter(MoreDynamicsModelPF,  # particle filter model class
            os.path.join(input_dir, "pedestrians.avi"),  # input video
            get_template_rect(os.path.join(input_dir, "pedestrians.txt")),  # suggested template window (dict)
            {
                'template': os.path.join(output_dir, 'ps7-4-b-1.png'),
                0: os.path.join(output_dir, 'ps7-4-b-1a.png'),
                40: os.path.join(output_dir, 'ps7-4-b-1b.png'),
                100: os.path.join(output_dir, 'ps7-4-b-1c.png'),
                133: os.path.join(output_dir, 'ps7-4-b-1d.png'),
                240: os.path.join(output_dir, 'ps7-4-b-1e.png')
            },  # frames to save, mapped to filenames, and 'template' if desired
            num_particles=20, sigma_MSE=30, dynamics_error=2, 
            ist_bin=8, scale_error=0.005, scale_min=0.25, init_method=init_method)

if __name__ == "__main__":
    main()
