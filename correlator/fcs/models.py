import numpy as np

def diffusion3D(r, a):
    return 1 / ((1+r)*np.sqrt(1+r/a**2))

def diffusion2D(r):
    return 1 / (1 + r)

def triplet_decay(r, F):
    return 1 + np.exp(-r)* F/(1-F)

def triplet_diffusion(t, T1, F_trip, T_trip, a, n=1):
    return triplet_decay(t/T_trip, F_trip) * diffusion3D(t/T1, a) / n

def flow_decay(r):
    return np.exp(-r)**2