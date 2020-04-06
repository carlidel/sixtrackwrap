from numba import jit, njit
import numpy as np
import sixtracklib as st
from conversion import convert_norm_to_physical, convert_physical_to_norm
import warnings

def track_particles(x, px, y, py, n_turns):
    """Wrap Sixtracklib and track the particles requested
    
    Parameters
    ----------
    x : ndarray
        initial conditions
    px : ndarray
        initial conditions
    y : ndarray
        initial conditions
    py : ndarray
        initial conditions
    n_turns : unsigned int
        number of turns to perform
    
    Returns
    -------
    particles object
        Sixtracklib particles object
    """    
    assert len(x) == len(px)
    assert len(x) == len(py)
    assert len(x) == len(y)

    particles = st.Particles.from_ref(
        num_particles=len(x), p0c=6.5e12)

    particles.x += x
    particles.px += px
    particles.y += y
    particles.py += py

    lattice = st.Elements.fromfile('data/beam_elements.bin')
    cl_job = st.TrackJob(lattice, particles, device="opencl:0.0")

    status = cl_job.track_until(n_turns)
    cl_job.collect_particles()
    return particles


def full_track_particles(radiuses, alpha, theta1, theta2, n_turns):
    """Complete tracking of particles for the given number of turns
    
    Parameters
    ----------
    radiuses : ndarray
        initial conditions
    alpha : ndarray
        initial conditions
    theta1 : ndarrayq
        initial conditions
    theta2 : ndarray
        initial conditions
    n_turns : unsigned int
        number of turns to perform
    
    Returns
    -------
    tuple
        (r, alpha, theta1, theta2), shape = (initial conditios, n turns)
    """    
    x, px, y, py = polar_to_cartesian(radiuses, alpha, theta1, theta2)
    x, px, y, py = convert_norm_to_physical(x, px, y, py)

    particles = st.Particles.from_ref(
        num_particles=len(x), p0c=6.5e12)

    particles.x += x
    particles.px += px
    particles.y += y
    particles.py += py

    lattice = st.Elements.fromfile('data/beam_elements.bin')
    cl_job = st.TrackJob(lattice, particles, device="opencl:0.0")

    data_r = np.empty((len(x), n_turns))
    data_a = np.empty((len(x), n_turns))
    data_th1 = np.empty((len(x), n_turns))
    data_th2 = np.empty((len(x), n_turns))

    for i in range(n_turns):
        status = cl_job.track_until(1)
        cl_job.collect_particles()
        
        t_x, t_px, t_y, t_py = convert_physical_to_norm(particles.x, particles.px, particles.y, particles.py)
        data_r[:, i], data_a[:, i], data_th1[:, i], data_th2[:, i] = cartesian_to_polar(t_x, t_px, t_y, t_py)
        
    return data_r, data_a, data_th1, data_th2


@njit
def polar_to_cartesian(radius, alpha, theta1, theta2):
    """Convert polar coordinates to cartesian coordinates
    
    Parameters
    ----------
    radius : ndarray
        ipse dixit
    alpha : ndarray
        ipse dixit
    theta1 : ndarray
        ipse dixit
    theta2 : ndarray
        ipse dixit
    
    Returns
    -------
    tuple of ndarrays
        x, px, y, py
    """    
    x = radius * np.cos(alpha) * np.cos(theta1)
    px = radius * np.cos(alpha) * np.sin(theta1)
    y = radius * np.sin(alpha) * np.cos(theta2)
    py = radius * np.sin(alpha) * np.sin(theta2)
    return x, px, y, py


@njit
def cartesian_to_polar(x, px, y, py):
    """Convert cartesian coordinates to polar coordinates
    
    Parameters
    ----------
    x : ndarray
        ipse dixit
    px : ndarray
        ipse dixit
    y : ndarray
        ipse dixit
    py : ndarray
        ipse dixit
    
    Returns
    -------
    tuple of ndarrays
        r, alpha, theta1, theta2
    """    
    r = np.sqrt(np.power(x, 2) + np.power(y, 2) +
                np.power(px, 2) + np.power(py, 2))
    theta1 = np.arctan2(px, x)
    theta2 = np.arctan2(py, y)
    alpha = np.arctan2(np.sqrt(y * y + py * py),
                       np.sqrt(x * x + px * px))
    return r, alpha, theta1, theta2


class radial_provider(object):
    """Base class for managing coordinate system on radiuses"""
    def __init__(self, alpha, theta1, theta2, dr, starting_step):
        """Init radial provider class
        
        Parameters
        ----------
        object : self
            base class for managing coordinate system on radiuses
        alpha : ndarray
            angles to consider
        theta1 : ndarray
            angles to consider
        theta2 : ndarray
            angles to consider
        dr : float
            radial step
        starting_step : unsiged int
            starting step point
        """        
        assert starting_step >= 0
        self.alpha = alpha
        self.theta1 = theta1
        self.theta2 = theta2
        self.dr = dr
        self.starting_step = starting_step

        self.count = 0
        self.active = True

    def get_positions(self, n_pos):
        """Get coordinates and update count
        
        Parameters
        ----------
        n_pos : unsigned int
            number of coordinates to gather
        
        Returns
        -------
        tuple of ndarrays
            (radius, alpha, theta1, theta2)
        """        
        assert n_pos > 0
        assert self.active
        a = np.ones(n_pos) * self.alpha
        th1 = np.ones(n_pos) * self.theta1
        th2 = np.ones(n_pos) * self.theta2
        r = np.linspace(self.starting_step + self.count, self.starting_step +
                        self.count + n_pos, n_pos, endpoint=False) * self.dr
        
        self.count += n_pos

        return r, a, th1, th2

    def peek_positions(self, n_pos):
        """Get coordinates WITHOUT updating the count
        
        Parameters
        ----------
        n_pos : unsigned int
            number of coordinates to gather
        
        Returns
        -------
        tuple of ndarrays
            (radius, alpha, theta1, theta2)
        """
        assert n_pos > 0
        assert self.active
        a = np.ones(n_pos) * self.alpha
        th1 = np.ones(n_pos) * self.theta1
        th2 = np.ones(n_pos) * self.theta2
        r = np.linspace(self.starting_step + self.count, self.starting_step +
                        self.count + n_pos, n_pos, endpoint=False) * self.dr

        return r, a, th1, th2

    def reset(self):
        """Reset the count and the status
        """        
        self.active = True
        self.count = 0



class radial_scanner(object):
    def __init__(self, alpha, theta1, theta2, dr, starting_step=0):
        """Init a radial scanner object
        
        Parameters
        ----------
        object : radial scanner
            wrapper for doing a proper radial scanning
        alpha : ndarray
            angles to consider
        theta1 : ndarray
            angles to consider
        theta2 : ndarray
            angles to consider
        dr : float
            radial step
        starting_step : int, optional
            starting step, by default 0
        """        
        self.alpha = alpha
        self.theta1 = theta1
        self.theta2 = theta2
        self.dr = dr
        self.starting_step = starting_step

        self.radiuses = [radial_provider(alpha[i], theta1[i], theta2[i], dr, starting_step) for i in range(len(alpha))]
        self.steps = [np.array([]) for i in range(len(alpha))]
        self.n_elements = len(alpha)

    def scan(self, max_turns, min_turns, batch_size=10e4):
        """Perform a radial scanning
        
        Parameters
        ----------
        max_turns : unsigned int
            max number of turns to perform
        min_turns : unsigned int
            minimum number of turns to perform
        batch_size : unsigned int, optional
            batch size for parallel computing (OpenCL support), by default 10e4
        
        Returns
        -------
        ndarray
            step informations (array of arrays)
        """        
        turns_to_do = max_turns
        while True:
            n_active = 0
            for radius in self.radiuses:
                if radius.active:
                    n_active += 1
            
            if n_active == 0:
                return self.steps

            if batch_size % n_active == 0:
                sample_size = batch_size // n_active
            else:
                sample_size = (batch_size // n_active) + 1

            r = np.array([])
            a = np.array([])
            th1 = np.array([])
            th2 = np.array([])

            for radius in self.radiuses:
                if radius.active:
                    t_r, t_a, t_th1, t_th2 = radius.get_positions(sample_size)
                    r = np.concatenate((r, t_r))
                    a = np.concatenate((a, t_a))
                    th1 = np.concatenate((th1, t_th1))
                    th2 = np.concatenate((th2, t_th2))

            x, px, y, py = polar_to_cartesian(r, a, th1, th2)
            x, px, y, py = convert_norm_to_physical(x, px, y, py)
            particles = track_particles(x, px, y, py, turns_to_do)
            turns = particles.at_turn
            
            i = 0
            turns_to_do = 0
            for j, radius in enumerate(self.radiuses):
                if radius.active:
                    r_turns = turns[i * sample_size : (i + 1) * sample_size]
                    if r_turns[-1] < min_turns:
                        radius.active = False
                    self.steps[j] = np.concatenate((self.steps[j], r_turns))
                    turns_to_do = max(turns_to_do, r_turns[-1])
                    i += 1

    def extract_DA(self, sample_list):
        """Gather DA radial data from the step data
        
        Parameters
        ----------
        sample_list : ndarray
            values to consider
        
        Returns
        -------
        ndarray
            radial values (n_elements, sample_list)
        """        
        values = np.empty((self.n_elements, sample_list))
        for i in range(self.n_elements):
            for j, sample in enumerate(sample_list):
                values[i, j] = np.argmin(self.steps[i] > sample) - 1
                if values[i, j] < 0:
                    warnings.warn("Warning: you entered a too high/low sample value.")
                    values[i, j] = 0
                else:
                    values[i, j] = (values[i, j] + self.starting_step) * self.dr
        return values


