import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from pyjacket.filetools import FileManager

def model(t: np.ndarray, N, g_inf, tau_trip, F_trip, tau, a):
    """Triplet state + diffusion"""
    return triplet_state(t/tau_trip, F_trip)*diffusion(t/tau, a)/N + g_inf

def diffusion(r: np.ndarray, a):
    return 1 / ((1+r)*(1+a**(-2)*r)**0.5)

def triplet_state(r, F_trip):
    return (1 - F_trip + F_trip*np.exp(-r)) / (1 - F_trip)



# Fit a single ACF curve to a model



# == Load ACF
fm = FileManager(
    r'D:\Data\FCS',
    r'D:\Analysis\FCS',
    rel_path=r'20241120\01_R110',
    CSV_SEP=',',
)
data = fm.read_csv(r'01_R110_3_10_1_1.pt3', folder='csv', dst=True)

# == Slice the ACF
rng = (1e-3 <= data.tau) & (data.tau < 1e+3)
tau = data.tau[rng].to_numpy()
acf = data.acf[rng].to_numpy()

# == Initial guess
F1 = 1
N = 12.76
G_inf = 1
Ttrip = 2.70e-3
Ftrip = 0.139
T1 = 2.82e-2
a = 8.0

p0 = (N, G_inf, Ttrip, Ftrip, T1, a)
guess = model(tau, *p0)

# == Visualize ACF
if True:
    plt.plot(tau, acf)
    plt.plot(tau, guess, 'k--')

# == Fitting
if True:
    popt, pcov = curve_fit(model, tau, acf, p0=p0, method='lm')
    fit = model(tau, *popt)

    plt.plot(tau, fit, 'k-')

print(f"{popt = }")

# == Visualization
plt.xscale('log')
plt.show()


