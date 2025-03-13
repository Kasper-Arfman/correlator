import numpy as np
from scipy.optimize import nnls


def second_derivative_matrix(n: int):
    M = np.zeros((n, n))
    for i in range(1, n-1):
        M[i, i-1] = -1
        M[i, i]   =  2
        M[i, i+1] = -1
    return M

def tikhonov_Phillips_reg(A: np.ndarray, alpha: float, b: np.ndarray, w):
    """
    Transform a dataset:

    g(t) => P(tau)

    This is described by:

    g(t) = sum_tau P(tau) * f(t, tau)

    here, f(t, tau) is contribution of a single tau component.





    Solve x / minimize ||w(Ax - b)|| + alpha||Mx|| 

    INPUTS:
    -------
    A (array[n_t, n_tau]): exp(-t/tau): model to the data (such that b_prediction = Ax)

    alpha (float): regularization parameter

    b (array[n_t]): correlation y-data

    W (array[n_t]): weights for each timepoint

    METHODS
    -------
    b is the vector with the measurements
    x is the unknown vector we want to estimate
    W is the weights vectors
    M is the second order derivative matrix

    Moreover, we add a last equation to the system such that sum(x) equals 1!
                and we force the initial and last values to be equal to 0!

    This function is was based on a 
    Quora answer (https://scicomp.stackexchange.com/questions/10671/tikhonov-regularization-in-the-non-negative-least-square-nnls-pythonscipy) 
    given by Dr. Brian Borchers (https://scicomp.stackexchange.com/users/2150/brian-borchers)

    RETURNS
    -------
     - x: the hydrodynamic radius distribution
     - residualNorm (scalar): residual error (penalty due to fit accuracy and smoothness)
     - penalty norm: penalty due to smoothness alone. For diagnostic purposes
    """
    # == Arrays for constraints
    n_t, n_tau = A.shape
    # print(f"{A.shape = }")
    normalized = np.ones(n_tau)                    # sum(P) = 1
    first_tau = np.zeros(n_tau); first_tau[0] = 1  # P( 0) = 0
    last_tau = np.flip(first_tau)                  # P(-1) = 0

    # == Extend data to implement constraint
    w = np.append(w, np.array([1e3, 1e3, 1e3])) # weight to force the initial and last values equal to 0, and the sum of contributions equal to 1
    A = np.vstack([A, normalized, first_tau, last_tau])
    b = np.append(b, np.array([1,0,0]))
    n_t, n_tau = A.shape

    # == Implement weights
    w = np.sqrt(w)
    A = w.reshape(-1, 1) * A # Fidelity term
    b = w * b

    # == Second order derivative matrix
    M = np.zeros((n_tau, n_tau))
    for i in range(1, n_tau-1):
        M[i, i-1] = -1
        M[i, i]   =  2
        M[i, i+1] = -1

    # == Rewrite the problem in a form nnls can solve (Ax-b)
    # - see Quora answer in function description
    A      = np.concatenate([A, np.sqrt(alpha) * M], axis=0)
    b      = np.concatenate([b, np.zeros(n_tau)]).flatten()

    # print(f"{A.shape = }")
    # exit()

    x, residualNorm   = nnls(A, b)  # Solves ||Ax - b||2
    penaltyNorm = np.linalg.norm(M.dot(x),2)
    return x, residualNorm, penaltyNorm


def fit_regularized(A: np.ndarray, b: np.ndarray, alpha=float, w:np.ndarray=None):
    # == Implement weights
    n_t, n_tau = A.shape
    w = np.sqrt(w) if w is not None else np.ones_like(b)
    A = w.reshape(-1, 1) * A
    b = w * b

    # == Second order derivative matrix
    M = second_derivative_matrix(n_tau)

    # == Rewrite the problem in a form nnls can solve (Ax-b)
    A      = np.concatenate([A, np.sqrt(alpha) * M], axis=0)
    b      = np.concatenate([b, np.zeros(n_tau)]).flatten()

    x, residual   = nnls(A, b)  # Solves ||Ax - b||2
    residual_smoothness = np.linalg.norm(M.dot(x),2)
    return x, residual, residual_smoothness