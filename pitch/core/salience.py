"""
| Description: libf0 salience-based F0 estimation implementation
| Author: Sebastian Rosenzweig, Simon Schwär, Meinard Müller
| License: The MIT license, https://opensource.org/licenses/MIT
| This file is part of libf0.
"""
import numpy as np
from librosa import stft
from scipy import ndimage, linalg
from numba import njit


def salience(x, Fs=22050, N=2048, H=256, F_min=55.0, F_max=1760.0, R=10.0, num_harm=10, freq_smooth_len=11,
             alpha=0.9, gamma=0.0, constraint_region=None, tol=5, score_low=0.01, score_high=1.0):
    """
    Implementation of a salience-based F0-estimation algorithm using pitch contours, inspired by Melodia.

    .. [#] Justin Salamon and Emilia Gómez,
       "Melody Extraction From Polyphonic Music Signals Using Pitch Contour Characteristics."
       IEEE Transactions on Audio, Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, Aug. 2012.

    Parameters
    ----------
    x : ndarray
        Audio signal
    Fs : int
        Sampling rate
    N : int
        Window size
    H : int
        Hop size
    F_min : float or int
        Minimal frequency
    F_max : float or int
        Maximal frequency
    R : int
        Frequency resolution given in cents
    num_harm : int
        Number of harmonics (Default value = 10)
    freq_smooth_len : int
        Filter length for vertical smoothing (Default value = 11)
    alpha : float
        Weighting parameter for harmonics (Default value = 0.9)
    gamma : float
        Logarithmic compression factor (Default value = 0.0)
    constraint_region : None or ndarray
        Constraint regions, row-format: (t_start_sec, t_end_sec, f_start_hz, f_end,hz)
        (Default value = None)
    tol : int
        Tolerance parameter for transition matrix (Default value = 5)
    score_low : float
        Score (low) for transition matrix (Default value = 0.01)
    score_high : float
        Score (high) for transition matrix (Default value = 1.0)

    Returns
    -------
    f0 : ndarray
        Estimated F0-trajectory
    T_coef: ndarray
        Time axis
    sal: ndarray
        Salience value of estimated F0
    
    See also
    --------
    [FMP] Notebook: C8/C8S2_SalienceRepresentation.ipynb
    """

    # compute salience representation via instantaneous frequency and harmonic summation
    Z, F_coef_hertz = compute_salience_rep(x, Fs, N=N, H=H, F_min=F_min, F_max=F_max, R=R,
                                           num_harm=num_harm, freq_smooth_len=freq_smooth_len,
                                           alpha=alpha, gamma=gamma)

    # compute trajectory via dynamic programming
    T_coef = (np.arange(Z.shape[1]) * H) / Fs
    index_CR = compute_trajectory_cr(Z, T_coef, F_coef_hertz, constraint_region,
                                     tol=tol, score_low=score_low, score_high=score_high)

    traj = F_coef_hertz[index_CR]
    traj[index_CR == -1] = 0

    # compute salience value
    Z_max = np.max(Z, axis=0)
    Z_norm = np.divide(Z, np.ones((Z.shape[0], 1)) * Z_max)
    sal = Z_norm[index_CR, np.arange(Z.shape[1])]
    sal[traj == 0] = 0

    return traj, T_coef, sal


def compute_salience_rep(x, Fs, N, H, F_min, F_max, R, num_harm, freq_smooth_len, alpha, gamma):
    """
    Compute salience representation [FMP, Eq. (8.56)]

    Parameters
    ----------
    x : ndarray
        Audio signal
    Fs : int
        Sampling rate
    N : int
        Window size
    H : int
        Hop size
    F_min : float or int
        Minimal frequency
    F_max : float or int
        Maximal frequency
    R : int
        Frequency resolution given in cents
    num_harm : int
        Number of harmonics
    freq_smooth_len : int
        Filter length for vertical smoothing
    alpha : float
        Weighting parameter for harmonics
    gamma : float
        Logarithmic compression factor

    Returns
    -------
    Z : ndarray
        Salience representation
    F_coef_hertz : ndarray
        Frequency axis in Hz

    See also
    --------
    [FMP] Notebook: C8/C8S2_SalienceRepresentation.ipynb
    """

    X = stft(x, n_fft=N, hop_length=H, win_length=N, pad_mode='constant')
    Y_LF_IF_bin, F_coef_hertz = compute_y_lf_if_bin_eff(X, Fs, N, H, F_min, F_max, R)
    
    # smoothing
    Y_LF_IF_bin = ndimage.convolve1d(Y_LF_IF_bin, np.hanning(freq_smooth_len), axis=0, mode='constant')
    
    Z = compute_salience_from_logfreq_spec(Y_LF_IF_bin, R, n_harmonics=num_harm, alpha=alpha, beta=1, gamma=gamma)
    return Z, F_coef_hertz


def compute_y_lf_if_bin_eff(X, Fs, N, H, F_min, F_max, R):
    """
    Binned Log-frequency Spectrogram with variable frequency resolution based on instantaneous frequency,
    more efficient implementation than FMP

    Parameters
    ----------
    X : ndarray
        Complex spectrogram
    Fs : int
        Sampling rate in Hz
    N : int
        Window size
    H : int
        Hop size
    F_min : float or int
        Minimal frequency
    F_max : float or int
        Maximal frequency
    R : int
        Frequency resolution given in cents

    Returns
    -------
    Y_LF_IF_bin : ndarray
        Binned log-frequency spectrogram using instantaneous frequency (shape: [freq, time])
    F_coef_hertz : ndarray
        Frequency axis in Hz
    """

    # calculate number of bins on log frequency axis
    B = frequency_to_bin_index(F_max, R, F_min) + 1

    # center frequencies of the final bins
    F_coef_hertz = F_min * np.power(2, (np.arange(0, B) * R / 1200))

    # calculate heterodyned phase increment (hpi)
    k = np.arange(X.shape[0]).reshape(-1, 1)
    omega = 2 * np.pi * k / N  # center frequency for each bin in rad
    hpi = (np.angle(X[:, 1:]) - np.angle(X[:, 0:-1])) - omega * H

    # reduce hpi to -pi:pi range
    # this is much faster than using the modulo function below, but gives the same result
    # hpi = np.mod(hpi + np.pi, 2 * np.pi) - np.pi
    hpi = hpi - 2 * np.pi * (np.around((hpi / (2 * np.pi)) + 1) - 1)

    # calculate instantaneous frequencies in Hz
    inst_f = (omega + hpi / H) * Fs / (2 * np.pi)
    # repeat the first time frame to match dimensions of X
    inst_f = np.hstack((np.copy(inst_f[:, 0]).reshape(-1, 1), inst_f))

    # mask frequencies that are not relevant
    mask = np.logical_and(inst_f >= F_min, inst_f < F_max)
    inst_f *= mask
    # set 0 to nan, so it does stay at nan in the bin assignment calculation
    inst_f[np.where(inst_f == 0)] = np.nan

    # find which inst_f values belong to which bin
    bin_assignment = frequency_to_bin_index(inst_f, R, F_min)
    # we map the discarded values to an extra bin that we remove before returning the binned spectrogram
    bin_assignment[np.where(np.isnan(inst_f))] = B

    # perform binning on power spectrogram for each time frame separately
    Y = np.abs(X) ** 2
    Y_LF_IF_bin = np.zeros((B+1, Y.shape[1]))
    for t in range(Y.shape[1]):
        np.add.at(Y_LF_IF_bin[:, t], bin_assignment[:, t], Y[:, t])

    return Y_LF_IF_bin[:B, :], F_coef_hertz


def compute_salience_from_logfreq_spec(lf_spec, R, n_harmonics, alpha, beta, gamma, harmonic_win_len=11):
    """
    Compute salience representation using harmonic summation following [1]

    [1] J. Salamon and E. Gomez,
        "Melody Extraction From Polyphonic Music Signals Using Pitch Contour Characteristics."
        IEEE Transactions on Audio, Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, Aug. 2012.

    Parameters
    ----------
    lf_spec : ndarray
        (F, T) log-spectrogram
    R : int
        Frequency resolution given in cents
    n_harmonics : int
        Number of harmonics
    alpha : float
        Weighting parameter for harmonics
    beta : float
        Compression parameter for spectrogram magnitudes
    gamma : float
        Magnitude threshold
    harmonic_win_len : int
        Length of a frequency weighting window in bins

    Returns
    -------
    Z : ndarray
        (F, T) salience representation of the input spectrogram
    """

    # magnitude thresholding and compression
    eps = np.finfo(np.float32).eps
    threshold_mask = (20 * np.log10(lf_spec/np.max(lf_spec) + eps)) < gamma
    lf_spec = lf_spec**beta * threshold_mask

    # compute window
    max_diff_bins = harmonic_win_len // 2
    window = np.cos(np.linspace(-1, 1, 2*max_diff_bins+1)*np.pi/2)**2  # cosine^2 window

    # compute indices of harmonics
    harmonics = np.round(np.log2(np.arange(1, n_harmonics + 1)) * 1200 / R).astype(int)
    weighting_vec = np.zeros((lf_spec.shape[0] + max_diff_bins))

    # compute weights
    for idx, h in enumerate(harmonics):
        if h+harmonic_win_len > len(weighting_vec):
            break  # we reached the maximum length available
        weighting_vec[h:h+harmonic_win_len] += window * alpha**idx

    # correlate lf_spec with the weighting vector on the frequency axis
    Z = ndimage.correlate1d(lf_spec, weighting_vec[:],
                            axis=0, mode='constant', cval=0, origin=-len(weighting_vec)//2 + max_diff_bins)

    # magnitude thresholding and compression
    threshold_mask = (20 * np.log10(Z / np.max(Z) + eps)) < gamma
    Z = Z ** beta * threshold_mask

    return Z


def define_transition_matrix(B, tol=0, score_low=0.01, score_high=1.0):
    """
    Generate transition matrix for dynamic programming

    Parameters
    ----------
    B : int
        Number of bins
    tol : int
        Tolerance parameter for transition matrix (Default value = 0)
    score_low : float
        Score (low) for transition matrix (Default value = 0.01)
    score_high : float
        Score (high) for transition matrix (Default value = 1.0)
    
    Returns
    -------
    T : ndarray
        (B, B) Transition matrix

    See also
    --------
    [FMP] Notebook: C8/C8S2_FundFreqTracking.ipynb
    """

    col = np.ones((B,)) * score_low
    col[0:tol+1] = np.ones((tol+1, )) * score_high
    T = linalg.toeplitz(col)
    return T


@njit
def compute_trajectory_dp(Z, T):
    """
    Trajectory tracking using dynamic programming

    Parameters
    ----------
    Z : ndarray
        Salience representation
    T : ndarray
        Transisition matrix
    
    Returns
    -------
    eta_DP : ndarray
        Trajectory indices

    See also
    --------
    [FMP] Notebook: C8/C8S2_FundFreqTracking.ipynb
    """

    B, N = Z.shape
    eps_machine = np.finfo(np.float32).eps
    Z_log = np.log(Z + eps_machine)
    T_log = np.log(T + eps_machine)

    E = np.zeros((B, N))
    D = np.zeros((B, N))
    D[:, 0] = Z_log[:, 0]

    for n in np.arange(1, N):
        for b in np.arange(0, B):
            D[b, n] = np.max(T_log[b, :] + D[:, n-1]) + Z_log[b, n]
            E[b, n-1] = np.argmax(T_log[b, :] + D[:, n-1])

    # backtracking
    eta_DP = np.zeros(N)
    eta_DP[N-1] = int(np.argmax(D[:, N-1]))

    for n in np.arange(N-2, -1, -1):
        eta_DP[n] = E[int(eta_DP[n+1]), n]

    return eta_DP.astype(np.int64)


def compute_trajectory_cr(Z, T_coef, F_coef_hertz, constraint_region=None,
                          tol=5, score_low=0.01, score_high=1.0):
    """
    Trajectory tracking with constraint regions
    Notebook: C8/C8S2_FundFreqTracking.ipynb

    Parameters
    ----------
    Z  : ndarray
        Salience representation
    T_coef : ndarray
        Time axis
    F_coef_hertz : ndarray
        Frequency axis in Hz
    constraint_region : ndarray or None
        Constraint regions, row-format: (t_start_sec, t_end_sec, f_start_hz, f_end_hz)
        (Default value = None)
    tol : int
        Tolerance parameter for transition matrix (Default value = 5)
    score_low : float
        Score (low) for transition matrix (Default value = 0.01)
    score_high : float
        Score (high) for transition matrix (Default value = 1.0)
    
    Returns
    -------
    eta : ndarray
        Trajectory indices, unvoiced frames are indicated with -1

    See also
    --------
    [FMP] Notebook: C8/C8S2_FundFreqTracking.ipynb
    """

    # do tracking within every constraint region
    if constraint_region is not None:
        # initialize contour, unvoiced frames are indicated with -1
        eta = np.full(len(T_coef), -1)

        for row_idx in range(constraint_region.shape[0]):
            t_start = constraint_region[row_idx, 0]  # sec
            t_end = constraint_region[row_idx, 1]  # sec
            f_start = constraint_region[row_idx, 2]  # Hz
            f_end = constraint_region[row_idx, 3]  # Hz

            # convert start/end values to indices
            t_start_idx = np.argmin(np.abs(T_coef - t_start))
            t_end_idx = np.argmin(np.abs(T_coef - t_end))
            f_start_idx = np.argmin(np.abs(F_coef_hertz - f_start))
            f_end_idx = np.argmin(np.abs(F_coef_hertz - f_end))

            # track in salience part
            cur_Z = Z[f_start_idx:f_end_idx+1, t_start_idx:t_end_idx+1]
            T = define_transition_matrix(cur_Z.shape[0], tol=tol,
                                         score_low=score_low, score_high=score_high)
            cur_eta = compute_trajectory_dp(cur_Z, T)

            # fill contour
            eta[t_start_idx:t_end_idx+1] = f_start_idx + cur_eta
    else:
        T = define_transition_matrix(Z.shape[0], tol=tol, score_low=score_low, score_high=score_high)
        eta = compute_trajectory_dp(Z, T)

    return eta


def frequency_to_bin_index(F, R, F_ref):
    """
        Binning function with variable frequency resolution
        Note: Indexing starts with 0 (opposed to [FMP, Eq. (8.49)])

    Parameters
    ----------
    F : float or ndarray
        Frequency in Hz
    R : float
        Frequency resolution in cents (Default value = 10.0)
    F_ref : float
        Reference frequency in Hz (Default value = 55.0)
    
    Returns
    -------
        bin_index (int): Index for bin (starting with index 0)

    See also
    --------
    [FMP] Notebook: C8/C8S2_SalienceRepresentation.ipynb
    """
    bin_index = np.floor((1200 / R) * np.log2(F / F_ref) + 0.5).astype(np.int64)
    return bin_index
