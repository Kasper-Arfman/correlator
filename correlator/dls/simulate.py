import numpy as np
from scipy.constants import k

def decay(t, tau):
    return np.exp(-t/tau)

def bragg_factor(n, lamb, radians):
    q = 4*np.pi * n * np.sin(radians/2) / lamb
    return q

def tau2radius(tau, n, lamb, radians, T, eta):
    gamma = 1/tau
    D = gamma / bragg_factor(n, lamb, radians)**2
    r = k*T / (6 * np.pi * eta * D)
    return r

def radius2tau(r, n, lamb, radians, T, eta):
    D =  k*T / (6 * np.pi * eta * r)
    gamma = D * bragg_factor(n, lamb, radians)**2
    tau = 1 / gamma
    return tau