import pandas as pd
import numpy as np
from correlator.dls.dataset import DataSet
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 5]
pd.set_option('display.max_columns', 8)
pd.set_option('display.max_rows', 16)
pd.set_option('display.expand_frame_repr', False)


TEST_PATH = r'C:\Users\arfma005\Documents\GitHub\correlation-analysis\test_DLS.csv'
LASER_WAVELENGTH = 817e-9  # [m]
SCATTERING_ANGLE = 150  # [deg]
RADIUS_INTERVAL = (0.09, 1e6, 200)  # lb, ub, N  geometrically spaced

# == Load the correlation data
df = pd.read_csv(TEST_PATH)  # first column must be time data in microseconds.
d = DataSet(
    df=df,
    laser_wavelength=LASER_WAVELENGTH,
    scattering_angle=SCATTERING_ANGLE,
)

# == Discrete hydrodynamic radius distribution
d.set_fitting_space(*RADIUS_INTERVAL)   #  Discretize the decay rate space we will use for the fitting

# G2 correlation curve
if False:
    plt.xscale("log")
    plt.plot(d.time,d.g2[:,0],'bo',markersize=1)
    plt.plot(d.time,d.g2[:,1],'bo',markersize=1)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Second order autocorrelation")
    plt.show()

# G1 correlation curve
if True:
    plt.xscale("log")
    plt.plot(d.time,d.g1[:,0],'bo',markersize=1)
    plt.plot(d.time,d.g1[:,1],'bo',markersize=1)
    plt.xlabel("Time (seconds)")
    plt.ylabel("First order autocorrelation")
    plt.show()

d.fit()

# L-curve plot
if True:
    figure, axis = plt.subplots(1, 2)
    axis[0].plot(np.log(d.smoothness_and_fit_penalty[:,0]),np.log(d.smoothness_penalty[:,0]),'bo',markersize=1)
    axis[0].plot(np.log(d.smoothness_and_fit_penalty[:,0])[d.i_alpha[0]],
            np.log(d.smoothness_penalty[:,0])[d.i_alpha[0]],'x',color='red',markersize=10)
    axis[0].set_xlabel("Log(fidelity term)")
    axis[0].set_ylabel("Log(penalty term)")
    axis[1].plot(np.log(d.smoothness_and_fit_penalty[:,1]),np.log(d.smoothness_penalty[:,1]),'bo',markersize=1)
    axis[1].plot(np.log(d.smoothness_and_fit_penalty[:,1])[d.i_alpha[1]],
            np.log(d.smoothness_penalty[:,1])[d.i_alpha[1]],'x',color='red',markersize=10)
    axis[1].set_xlabel("Log(fidelity term)")
    axis[1].set_ylabel("Log(penalty term)")
    plt.show()

# Fitted g2 plot
if True:
    plt.xscale("log")

    plt.plot(d.time,d.g2_estimate[:,0],'red')
    plt.plot(d.time,d.g2_estimate[:,1],'red')
    plt.plot(d.time,d.g2[:,0],'bo',markersize=1)
    plt.plot(d.time,d.g2[:,1],'bo',markersize=1)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Second order autocorrelation")
    plt.show()

# Distributions plot
if True:
    plt.xscale("log")

    plt.plot(d.rh,d.rh_contrib[0],'blue')
    plt.plot(d.rh,d.rh_contrib[1],'orange')

    plt.xlabel("Hydrodynamic radius (nm)")
    plt.ylabel("Relative contribution")
    plt.show()
