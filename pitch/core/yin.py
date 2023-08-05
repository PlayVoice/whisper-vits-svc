"""
| Description: libf0 YIN implementation
| Contributors: Sebastian Rosenzweig, Simon Schwär, Edgar Suárez, Meinard Müller
| License: The MIT license, https://opensource.org/licenses/MIT
| This file is part of libf0.
"""
import numpy as np
from numba import njit


def yin(x, Fs=22050, N=2048, H=256, F_min=55.0, F_max=1760.0, threshold=0.15, verbose=False):
    """
    Implementation of the YIN algorithm.

    .. [#] Alain De Cheveigné and Hideki Kawahara.
        "YIN, a fundamental frequency estimator for speech and music."
        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.

    Parameters
    ----------
    x : ndarray [shape=(L, )], real - valued
        Audio signal
    Fs : int
        Sampling frequency
    N : int
        Window size
    H : int
        Hop size
    F_min : float
        Minimal frequency
    F_max : float
        Maximal frequency
    threshold : float
        Threshold for cumulative mean normalized difference function
    verbose : bool
        Switch to activate/deactivate status bar

    Returns
    -------
    f0 : ndarray
        Estimated F0-trajectory
    t : ndarray
        Time axis
    ap: ndarray
        Aperiodicity (indicator for voicing: the lower, the more reliable the estimate)
    """

    if F_min > F_max:
        raise Exception("F_min must be smaller than F_max!")

    if F_min < Fs/N:        
        raise Exception(f"The condition (F_min >= Fs/N) was not met. With Fs = {Fs}, N = {N} and F_min = {F_min} you have the following options: \n1) Set F_min >= {np.ceil(Fs/N)} Hz. \n2) Set N >= {np.ceil(Fs/F_min).astype(int)}. \n3) Set Fs <= {np.floor(F_min * N)} Hz.")

    x_pad = np.concatenate((np.zeros(N//2), x, np.zeros(N//2)))  # Add zeros for centered estimates
    M = int(np.floor((len(x_pad) - N) / H)) + 1  # Compute number of estimates that will be generated
    f0 = np.zeros(M)  # Estimated fundamental frequencies (0 for unspecified frames)
    t = np.arange(M)*H/Fs  # Time axis
    ap = np.zeros(M)  # Aperiodicity

    lag_min = max(int(np.ceil(Fs / F_max)), 1)  # lag of maximal frequency in samples
    lag_max = int(np.ceil(Fs / F_min))  # lag of minimal frequency in samples

    for m in range(M):
        if verbose:
            print(f"YIN Progress: {np.ceil(100*m/M).astype(int)}%", end='\r')
        # Take a frame from input signal
        frame = x_pad[m*H:m*H + N]

        # Cumulative Mean Normalized Difference Function
        cmndf = cumulative_mean_normalized_difference_function(frame, lag_max)

        # Absolute Thresholding
        lag_est = absolute_thresholding(cmndf, threshold, lag_min, lag_max, parabolic_interp=True)

        # Refine estimate by constraining search to vicinity of best local estimate (default: +/- 25 cents)
        tol_cents = 25
        lag_min_local = int(np.round(Fs / ((Fs / lag_est) * 2 ** (tol_cents/1200))))
        if lag_min_local < lag_min:
            lag_min_local = lag_min
        lag_max_local = int(np.round(Fs / ((Fs / lag_est) * 2 ** (-tol_cents/1200))))
        if lag_max_local > lag_max:
            lag_max_local = lag_max
        lag_new = absolute_thresholding(cmndf, threshold=np.inf, lag_min=lag_min_local, lag_max=lag_max_local,
                                        parabolic_interp=True)

        # Compute Fundamental Frequency Estimate
        f0[m] = Fs / lag_new

        # Compute Aperiodicity
        ap[m] = aperiodicity(frame, lag_new)

    return f0, t, ap


@njit
def cumulative_mean_normalized_difference_function(frame, lag_max):
    """
    Computes Cumulative Mean Normalized Difference Function (CMNDF).

    Parameters
    ----------
    frame : ndarray
        Audio frame
    lag_max : int
        Maximum expected lag in the CMNDF

    Returns
    -------
    cmndf : ndarray
        Cumulative Mean Normalized Difference Function
    """

    cmndf = np.zeros(lag_max+1)  # Initialize CMNDF
    cmndf[0] = 1
    diff_mean = 0

    for tau in range(1, lag_max+1):
        # Difference function
        diff = np.sum((frame[0:-tau] - frame[0 + tau:]) ** 2)
        # Iterative mean of the difference function
        diff_mean = diff_mean*(tau-1)/tau + diff/tau

        cmndf[tau] = diff / (diff_mean + np.finfo(np.float64).eps)

    return cmndf


def absolute_thresholding(cmndf, threshold, lag_min, lag_max, parabolic_interp=True):
    """
    Absolute thresholding:
    Set an absolute threshold and choose the smallest value of tau that gives a minimum of d' deeper than that
    threshold. If none is found, the global minimum is chosen instead.

    Parameters
    ----------
    cmndf : ndarray
        Cumulative Mean Normalized Difference Function
    threshold : float
        Threshold
    lag_min : float
        Minimal lag
    lag_max : float
        Maximal lag
    parabolic_interp : bool
        Switch to activate/deactivate parabolic interpolation

    Returns
    -------

    """

    # take shortcut if search range only allows for one possible lag
    if lag_min == lag_max:
        return lag_min

    # find local minima below absolute threshold in interval [lag_min:lag_max]
    local_min_idxs = (np.argwhere((cmndf[1:-1] < cmndf[0:-2]) & (cmndf[1:-1] < cmndf[2:]))).flatten() + 1
    below_thr_idxs = np.argwhere(cmndf[lag_min:lag_max] < threshold).flatten() + lag_min
    # numba compatible intersection of indices sets
    min_idxs = np.unique(np.array([i for i in local_min_idxs for j in below_thr_idxs if i == j]))

    # if no local minima below threshold are found, return global minimum
    if not min_idxs.size:
        return np.argmin(cmndf[lag_min:lag_max]) + lag_min

    # find first local minimum
    lag = np.min(min_idxs)  # choose first local minimum

    # Optional: Parabolic Interpolation of local minima
    if parabolic_interp:
        lag_corr, cmndf[lag] = parabolic_interpolation(cmndf[lag-1], cmndf[lag], cmndf[lag+1])
        lag += lag_corr

    return lag


@njit
def parabolic_interpolation(y1, y2, y3):
    """
    Parabolic interpolation of an extremal value given three samples with equal spacing on the x-axis.
    The middle value y2 is assumed to be the extremal sample of the three.

    Parameters
    ----------
    y1: f(x1)
    y2: f(x2)
    y3: f(x3)

    Returns
    -------
    x_interp: Interpolated x-value (relative to x3-x2)
    y_interp: Interpolated y-value, f(x_interp)
    """

    a = np.finfo(np.float64).eps + (y1 + y3 - 2 * y2) / 2
    b = (y3 - y1) / 2
    x_interp = -b / (2 * a)
    y_interp = y2 - (b ** 2) / (4 * a)

    return x_interp, y_interp


def aperiodicity(frame, lag_est):
    """
    Compute aperiodicity of given frame (serves as indicator for reliability or voicing detection).

    Parameters
    ----------
    frame : ndarray
        Frame
    lag_est : float
        Estimated lag

    Returns
    -------
    ap: float
        Aperiodicity (the lower, the more reliable the estimate)
    """

    lag_int = int(np.floor(lag_est))  # uncorrected period estimate
    frac = lag_est - lag_int  # residual

    # Pad frame to insure constant size
    frame_pad = np.concatenate((frame, np.flip(frame)))  # mirror padding

    # Shift frame by estimated period
    if frac == 0:
        frame_shift = frame_pad[lag_int:lag_int+len(frame)]
    else:
        # linear interpolation between adjacent shifts
        frame_shift = (1 - frac) * frame_pad[lag_int:lag_int+len(frame)] + \
                      frac * frame_pad[lag_int+1:lag_int+1+len(frame)]

    pwr = (np.mean(frame ** 2) + np.mean(frame_shift ** 2)) / 2  # average power over fixed and shifted frame
    res = np.mean((frame - frame_shift) ** 2) / 2  # residual power
    ap = res / (pwr + np.finfo(np.float64).eps)

    return ap
