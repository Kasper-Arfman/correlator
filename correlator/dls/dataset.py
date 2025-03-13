import numpy as np
import pandas as pd

from correlator.core.diffusion import diffusion_from_inverse_decay_rate, hydrodynamic_radius
from correlator.dls.dls_helpers import Bragg_wave_vector, estimate_beta, g1_from_g2, g2_finite_aproximation, get_contributios_prior, s_inverse_decay_rate
from correlator.models.dataset import AbstractDataset
from correlator.core.helpers import *

# class Dataset(AbstractDataset):

kb    = 1.380649e-23 # Joule / Kelvin

class DataSet(AbstractDataset):

    rh: np.ndarray  # Hydrodynamic radius
    rh_contrib: np.ndarray  # Hydrodynamic radius contributions

    s_space: np.ndarray
    ds: np.ndarray

    optimal_alpha: float
    i_alpha: int

    smoothness_and_fit_penalty: np.ndarray
    smoothness_penalty: np.ndarray
    residuals_g1: np.ndarray
    g2_estimate: np.ndarray

    def __init__(self, df: pd.DataFrame, laser_wavelength, scattering_angle, time_limit=1e8):

        # Autocorrelation data
        self.time, self.g2, sampleNames = parse_csv(df, i=1)  # exclude the first point
        self.time *= 1e-6  # us -> s




        self.sample_info = pd.DataFrame({"conditions":sampleNames,"read":1,"scan":1,"include":True})
        self.n_t, self.n_m  = self.g2.shape  # time_points, measurements

        # Experimental conditions
        self.temperature = 293     # [K]
        self.viscosity = 8.9e-4  # [Pa.s]
        self.refractive_index = 1.33    # [-]
        self.lambda0 = laser_wavelength  # [m]
        self.scattering_angle = scattering_angle / 180 * np.pi  # [rad]
        self.q = Bragg_wave_vector(self.lambda0, self.refractive_index, self.scattering_angle) 

        # self.q = 4*np.pi*self.refractive_index/self.lambda0 * np.sin(self.scattering_angle/2) * 1e9
        print(f"{self.q = }")

        # Fit conditions
        self.time_limit = time_limit
        self.alphas = (5**np.arange(-6,2,0.1, dtype=float))**2

        # G2 to G1
        self.beta_estimate = estimate_beta(self.g2, self.time) 
        self.g1 = np.column_stack([g1_from_g2(self.g2[:,i], self.beta_estimate[i]) for i in range(self.n_m)])

    def set_fitting_space(self, rh_min, rh_max, n):
        """
        Create the s (inverse of gamma decay rate) space that will be used for the fitting
        The limits are given by the minimum and maximum desired hydrodynamic radius (in nanometers)
        """
        n = int(n) # Convert n to integer type for the np.logspace function
        s_max = s_inverse_decay_rate(self.D(rh_max/1e9), self.q)
        s_min = s_inverse_decay_rate(self.D(rh_min/1e9), self.q)

        self.s_space = np.geomspace(s_min, s_max, n)
        self.ds = diffusion_from_inverse_decay_rate(self.s_space, self.q)
        self.rh = hydrodynamic_radius(self.ds, self.temperature, self.viscosity)*1e9  # [nm]
    
    def fit(self):
        self.fit_g1_many_alpha(self.alphas, self.time_limit)
        self.optimal_alpha = self.get_optimal_alpha()
        self.fit_g1(self.optimal_alpha, self.time_limit)
        self.update_g2()
    
    def fit_g1_many_alpha(self, alphas, time_limit):
        """
        Fit the g1 curve for a range of alphas.

        Store fit penalties and smoothness penaltys to choose the best alpha later
        """
        _t = self.time < (time_limit / 1e6)  # time boolean
        self.smoothness_and_fit_penalty, self.smoothness_penalty = [], []
        for alpha in alphas:
            _ , residual_norm, smoothness_norm = get_contributios_prior(
                self.g1[_t, :],
                self.time[_t],
                self.s_space, 
                alpha) 
            self.smoothness_and_fit_penalty.append(residual_norm)
            self.smoothness_penalty.append(smoothness_norm)
        self.smoothness_and_fit_penalty = np.array(self.smoothness_and_fit_penalty)
        self.smoothness_penalty  = np.array(self.smoothness_penalty)

    def fit_g1(self, alpha, timeLimit):
        """
        Fit the G1 curve using alpha and betaGuess

        Obtain initial estimates for the relative contributions
        Run after createFittingS_space() !
        timeLimit should be given in microseconds! Default time is 100 seconds (all the autocorrelation curve).
        alpha can be one value (same for all curves) or a list of values (one value per curve)
        """
        _t = self.time < (timeLimit / 1e6)

        # Return the fitted contributions and residuals of the first order autocorrelation function
        self.rh_contrib, self.residuals_g1, _   = get_contributios_prior(
            self.g1[_t],
            self.time[_t],
            self.s_space,
            alpha
            ) 

    def get_optimal_alpha(self):
        """
        Apply the triangle method to find the corner of the L-curve criteria en return the 'optimal' alpha for each curve
        """
        i_alpha = []
        # Iterate over the curves
        for idx in range(self.smoothness_and_fit_penalty.shape[1]):
            i_alpha.append(find_Lcurve_corner(self.smoothness_and_fit_penalty[:,idx],self.smoothness_penalty[:,idx]))
        self.i_alpha = i_alpha

        optimal_alpha = [self.alphas[idx] for idx in i_alpha]
        return optimal_alpha

    def update_g2(self):
        g2_estimate = []
        for idx in range(self.n_m):
 
            # check that we estimated the contributions!
            if len(self.rh_contrib[idx]) > 1:
                g2_estimate.append(g2_finite_aproximation(
                    1/self.s_space,
                    self.time,
                    self.beta_estimate[idx],
                    self.rh_contrib[idx]
                    ))

            else:
                # In the case we couldn't fit anything!
                g2_estimate.append(np.array(0))
        self.g2_estimate = np.column_stack(g2_estimate)

    def D(self, r):
        return kb*self.temperature / (6*np.pi*self.viscosity*r) # [m]