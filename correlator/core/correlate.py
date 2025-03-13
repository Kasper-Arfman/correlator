import numpy as np
import multipletau
import numba


@numba.njit
def _histogram(time, bins):
    return np.histogram(time, bins)

def bin_trace(time, bin_time, hertz=True):
    n_bins = int((time[-1] - time[0]) // bin_time)
    counts, edges = _histogram(time, n_bins)
    time = (edges[:-1] + edges[1:]) / 2
    if hertz: # Counts => Hz
        counts = counts / (time[1] - time[0])  
    return time, counts

def autocorrelate(binned_trace: np.ndarray, bin_time, m=16):
    tau, acf = multipletau.autocorrelate(binned_trace, m=m, normalize=True)[1:].T
    acf += 1
    tau *= bin_time
    return tau, acf