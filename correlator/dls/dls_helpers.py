import numpy as np
from correlator.core.fit.regularized import tikhonov_Phillips_reg


def g2_finite_aproximation(decay_rates, times, beta: np.ndarray, contributions):
    """
    Obtain the autocorrelation function based on decay rates and their
    relatives contributions

    beta is the intercept (g2 at time 0)
    """
    assert len(decay_rates) == len(contributions)

    g1 = np.array([np.sum(contributions*np.exp(-decay_rates*t)) for t in times])

    g2 = 1 + beta * (g1)**2
    return g2


def get_contributios_prior(g1: np.ndarray, time, s_space: np.ndarray, alpha, weights=None):
    """
    INPUT
    -----
    g1 (array[n, m])

    time (array[n])

    s_space to create the kernel for the Thinkohonov regularization function

    alpha

    weights


    RETURNS
    -------
    The estimated contribution of each decay rate (length defined by the s_space vector)

    """
    n_m = g1.shape[1]
    if not isinstance(alpha, list):
        alpha = [alpha]*n_m

    # Create a kernel containing s-space and time
    s_space = s_space.reshape((-1, 1))
    sM, tM = np.meshgrid(s_space, time, indexing='xy')
    A      = np.exp(-tM/sM)  # time / decay_time

    rh_contrib = []
    residuals     = []
    penaltyNorms  = []
    for i in range(n_m):

        # If there are nan values, analyze only a subset
        try:
            n_max      = np.min(np.argwhere(np.isnan(g1[:, i])))
        except:
            n_max      = len(g1[:, i])

        # Fit the g1 curve
        # Equal weights or custom weights
        w_i = np.ones(n_max) if weights is None else weights[:n_max, i]
        cont, residual, penaltyNorm = tikhonov_Phillips_reg(A[:n_max], alpha[i], g1[:n_max, i], w_i)

        rh_contrib.append(cont)
        residuals.append(residual)
        penaltyNorms.append(penaltyNorm)
    return rh_contrib, residuals, penaltyNorms


def Bragg_wave_vector(wavelength, refractive_index, scattering_angle):
    return 4 * np.pi * refractive_index * np.sin(scattering_angle/2) / wavelength


def s_inverse_decay_rate(D, q):
    return 1 / (D*(q**2)) # [s]


def g1_first_order_corr(s, t):
    return np.exp(-t/s) # [-]


def g2_second_order_corr(g1,beta):
    return 1 + beta * (g1)**2 # [-]


def g1_from_g2(g2: np.ndarray, beta, force_positive=False):
    if force_positive:
        g2 = np.where(g2>1, g2, 1)
    return np.sqrt( (g2-1) / beta) # [-]


def estimate_beta(g2, t) -> np.ndarray:
    npFit = np.polyfit(t[t < 5e-6], np.log(g2[t < 5e-6] - 1), 2)
    beta = np.exp(npFit[-1])
    return beta