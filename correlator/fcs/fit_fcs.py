import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from correlator.core.fit.regularized import fit_regularized, second_derivative_matrix
np.random.seed(1)

def diffusion(r, a):
    # r = t/tau
    return 1 / ((1+r)*np.sqrt(1+r/a**2))

def triplet(r, F):
    # tau: triplet lifetime
    return 1 + np.exp(-r)* F/(1-F)

def triplet_diffusion(t, t1, F_trip, t_trip, a):
    return triplet(t/t1, F_trip) * diffusion(t/t_trip, a)


# Simulate data
a = 5.5
t_1 = 0.050

F_trip = 0.15
t_trip = 0.003

t = np.geomspace(1e-6, 1e2, 400)
acf = triplet_diffusion(t, t_1, F_trip, t_trip, a)
acf = np.random.normal(1, 0.03, acf.size) * acf


tau = np.geomspace(1e-3, 1, 200)

if False:
    plt.plot(t, acf)
    plt.xscale('log')
    plt.show()



from scipy.optimize import least_squares


def build_kernel(t, tau, F_trip, T_trip, a):
    # Build the kernel matrix based on triplet_diffusion
    kernel = np.zeros((len(t), len(tau)))
    for j, t_j in enumerate(t):
        for i, tau_i in enumerate(tau):
            kernel[j, i] = triplet_diffusion(t_j, tau_i, F_trip, T_trip, a)
    return kernel


def fit_fcs(t, b, tau, alpha, F_trip_init, T_trip_init, a, w=None):
    # Initialize weights
    n_t = len(t)
    n_tau = len(tau)
    w = np.sqrt(w) if w is not None else np.ones_like(b)

    # Objective function for least_squares
    def objective(params):
        F_trip, T_trip = params[:2]  # Shared parameters
        P = params[2:]              # Contributions (P_i)
        
        # Build the kernel matrix for current F_trip and T_trip
        A = build_kernel(t, tau, F_trip, T_trip, a)
        
        # Weighted residuals for ||A*P - b||
        res_data = w * (A @ P - b)
        
        # Regularization term ||M*P||
        M = second_derivative_matrix(n_tau)
        res_reg = np.sqrt(alpha) * M @ P
        
        return np.concatenate([res_data, res_reg])

    # Initial guess for the parameters
    x0 = np.concatenate([[F_trip_init, T_trip_init], np.ones(n_tau)])

    # Bounds for parameters
    bounds_lower = [0, 0] + [0] * n_tau  # F_trip >= 0, T_trip >= 0, P_i >= 0
    bounds_upper = [1, np.inf] + [np.inf] * n_tau  # F_trip <= 1
    bounds = (bounds_lower, bounds_upper)

    # Solve the least squares problem
    print('fitting...')
    result = least_squares(objective, x0, bounds=bounds)
    print('done!')

    # Extract results
    x = result.x
    # F_trip_fit, T_trip_fit = result.x[:2]
    # P_fit = result.x[2:]
    # residual_norm = np.linalg.norm(objective(result.x), 2)
    return x


# Fit only t1
sM, tM = np.meshgrid(tau, t, indexing='xy')
A      = triplet_diffusion(tM, sM, F_trip, t_trip, a)


# x, _, _ = fit_regularized(A, acf, alpha=0.01)
x = fit_fcs(t, acf, tau, alpha=0.1, F_trip_init=0.5, T_trip_init=0.005, a=5.5)
# print(x)

print(x)

plt.plot(tau, x)
plt.xscale('log')
plt.show()