import numpy as np

"""This module contains the Twiss parameters and the materials necessary
to do conversion form normalized space to physical space and viceversa"""

m_p = 938e6  # [m_p] = 1 eV
c0 = 299792458.0  # [c0] = 1 m/s
Etot = 6.5e12  # [Etot] = 1 ev (6.5 Tev)

gamma = Etot / m_p
beta = np.sqrt(1 - 1 / gamma**2)

normed_emittance = 2.5e-6
emittance = normed_emittance / (beta * gamma)

beta_x = 119.6823009181
alpha_x = 2.2471670653
beta_y = 219.8117974618
alpha_y = -2.6665756601

x_0 = -0.1009048961e-04
vx_0 = -0.9409992992e-07
y_0 = -0.8981238025e-06
vy_0 = -0.3361671983e-07

s_0 = 0.1850381795e-05
vs_0 = 0.1778819208e-04

p_delta = 0.0


def convert_norm_to_physical(X, VX, Y, VY):
    """Convert normal coordinates to physical coordinates
    
    Parameters
    ----------
    X : ndarray
        X
    VX : ndarray
        X' basically
    Y : ndarray
        Y
    VY : ndarray
        Y' basically
    
    Returns
    -------
    tuple of ndarray
        (x, x', y, y')
    """    
    x = np.sqrt(emittance * beta_x) * X
    vx = np.sqrt(emittance) * (- alpha_x / np.sqrt(beta_x)
                               * X + 1 / np.sqrt(beta_x) * VX)
    y = np.sqrt(emittance * beta_y) * Y
    vy = np.sqrt(emittance) * (- alpha_y / np.sqrt(beta_y)
                               * Y + 1 / np.sqrt(beta_y) * VY)

    x += x_0
    vx += vx_0
    y += y_0
    vy += vy_0

    return x, vx, y, vy


def convert_physical_to_norm(x, vx, y, vy):
    """Convert physical coordinates to normal coordinates
    
    Parameters
    ----------
    x : ndarray
        x
    vx : ndarray
        x' basically
    y : ndarray
        y
    vy : ndarray
        y' basically
    
    Returns
    -------
    tuple of ndarray
        (x, x', y, y')
    """
    x -= x_0
    vx -= vx_0
    y -= y_0
    vy -= vy_0

    X = x / np.sqrt(emittance * beta_x)
    VX = 1 / np.sqrt(emittance) * (alpha_x / np.sqrt(beta_x)
                                   * x + np.sqrt(beta_x) * vx)

    Y = y / np.sqrt(emittance * beta_y)
    VY = 1 / np.sqrt(emittance) * (alpha_y / np.sqrt(beta_y)
                                   * y + np.sqrt(beta_y) * vy)

    return X, VX, Y, VY
