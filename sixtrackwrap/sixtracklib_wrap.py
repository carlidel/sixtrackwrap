from numba import jit, njit, prange
import numpy as np
from .conversion import convert_norm_to_physical, convert_physical_to_norm
import warnings
import os
import time
import pickle


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

try:
    import sixtracklib as st

    def track_particles(x, px, y, py, n_turns, opencl=True):
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
        opencl : bool (optional)
            use opencl backend (default: True)
        
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

        lattice = st.Elements.fromfile(os.path.join(
            os.path.dirname(__file__), 'data/beam_elements.bin'))
        if opencl:
            cl_job = st.TrackJob(lattice, particles, device="opencl:0.0")
        else:
            cl_job = st.TrackJob(lattice, particles)

        status = cl_job.track_until(n_turns)
        cl_job.collect_particles()
        return particles


    def full_track_particles(radiuses, alpha, theta1, theta2, n_turns, opencl=True):
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
        opencl : bool (optional)
            use opencl backend (default: True)
        
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

        lattice = st.Elements.fromfile(os.path.join(
            os.path.dirname(__file__), 'data/beam_elements.bin'))
        if opencl:
            cl_job = st.TrackJob(lattice, particles, device="opencl:0.0")
        else:
            cl_job = st.TrackJob(lattice, particles)

        data_r = np.empty((len(x), n_turns))
        data_a = np.empty((len(x), n_turns))
        data_th1 = np.empty((len(x), n_turns))
        data_th2 = np.empty((len(x), n_turns))

        for i in range(n_turns):
            status = cl_job.track_until(i)
            cl_job.collect_particles()
            # print(particles.at_turn)
            t_x, t_px, t_y, t_py = convert_physical_to_norm(
                particles.x, particles.px, particles.y, particles.py)
            data_r[:, i], data_a[:, i], data_th1[:, i], data_th2[:,
                                                                i] = cartesian_to_polar(t_x, t_px, t_y, t_py)

        return data_r, data_a, data_th1, data_th2
        
except ImportError:
    print("Sixtracklib module not found! Loading dummy functions!")

    def track_particles(x, px, y, py, n_turns, opencl=True):
        pass

    def full_track_particles(radiuses, alpha, theta1, theta2, n_turns, opencl=True):
        pass
