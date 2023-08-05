"""
| Description: libf0 yin implementation
| Contributors: Sebastian Rosenzweig, Simon Schwär, Edgar Suárez, Meinard Müller
| License: The MIT license, https://opensource.org/licenses/MIT
| This file is part of libf0.
"""
import numpy as np
from scipy.special import beta, comb  # Scipy library for binomial beta distribution
from scipy.stats import triang      # Scipy library for triangular distribution
from .yin import cumulative_mean_normalized_difference_function, parabolic_interpolation
from numba import njit


# pYIN estimate computation
def pyin(x, Fs=22050, N=2048, H=256, F_min=55.0, F_max=1760.0, R=10, thresholds=np.arange(0.01, 1, 0.01),
         beta_params=[1, 18], absolute_min_prob=0.01, voicing_prob=0.5):
    """
    Implementation of the pYIN F0-estimation algorithm.

    .. [#] Matthias Mauch and Simon Dixon.
        "PYIN: A fundamental frequency estimator using probabilistic threshold distributions".
        IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2014): 659-663.

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
    thresholds : ndarray
        Range of thresholds
    beta_params : tuple or list
        Parameters of beta-distribution in the form [alpha, beta]
    absolute_min_prob : float
        Prior for voice activity
    voicing_prob: float
        Prior for transition probability?
    Returns
    -------
    f0 : ndarray
        Estimated F0-trajectory
    t : ndarray
        Time axis
    conf : ndarray
        Confidence
    """

    if F_min > F_max:
        raise Exception("F_min must be smaller than F_max!")

    if F_min < Fs/N:        
        raise Exception(f"The condition (F_min >= Fs/N) was not met. With Fs = {Fs}, N = {N} and F_min = {F_min} you have the following options: \n1) Set F_min >= {np.ceil(Fs/N)} Hz. \n2) Set N >= {np.ceil(Fs/F_min).astype(int)}. \n3) Set Fs <= {np.floor(F_min * N)} Hz.")

    x_pad = np.concatenate((np.zeros(N // 2), x, np.zeros(N // 2)))  # Add zeros for centered estimates

    # Compute Beta distribution
    thr_idxs = np.arange(len(thresholds))
    beta_distr = comb(len(thresholds), thr_idxs) * beta(thr_idxs+beta_params[0],
                                                        len(thresholds)-thr_idxs+beta_params[1]) / beta(beta_params[0],
                                                                                                        beta_params[1])

    # YIN with multiple thresholds, yielding observation matrix
    B = int(np.log2(F_max / F_min) * (1200 / R))
    F_axis = F_min * np.power(2, np.arange(B) * R / 1200)  # for quantizing the estimated F0s
    O, rms, p_orig, val_orig = yin_multi_thr(x_pad, Fs=Fs, N=N, H=H, F_min=F_min, F_max=F_max, thresholds=thresholds,
                                             beta_distr=beta_distr, absolute_min_prob=absolute_min_prob, F_axis=F_axis,
                                             voicing_prob=voicing_prob)

    # Transition matrix, using triangular distribution used for pitch transition probabilities
    max_step_cents = 50  # Pitch jump can be at most 50 cents from frame to frame
    max_step = int(max_step_cents / R)
    triang_distr = triang.pdf(np.arange(-max_step, max_step+1), 0.5, scale=2*max_step, loc=-max_step)
    A = compute_transition_matrix(B, triang_distr)
    
    # HMM smoothing
    C = np.ones((2*B, 1)) / (2*B)  # uniform initialization
    f0_idxs = viterbi_log_likelihood(A, C.flatten(), O)  # libfmp Viterbi implementation
    
    # Obtain F0-trajectory
    F_axis_extended = np.concatenate((F_axis, np.zeros(len(F_axis))))
    f0 = F_axis_extended[f0_idxs]

    # Suppress low power estimates
    f0[0] = 0  # due to algorithmic reasons, we set the first value unvoiced
    f0[rms < 0.01] = 0

    # confidence
    O_norm = O[:, np.arange(O.shape[1])]/np.max(O, axis=0)
    conf = O_norm[f0_idxs, np.arange(O.shape[1])]

    # Refine estimates by choosing the closest original YIN estimate
    refine_estimates = True
    if refine_estimates:
        f0 = refine_estimates_yin(f0, p_orig, val_orig, Fs, R)

    t = np.arange(O.shape[1]) * H / Fs  # Time axis
    
    return f0, t, conf


@njit
def refine_estimates_yin(f0, p_orig, val_orig, Fs, tol):
    """
    Refine estimates using YIN CMNDF information.

    Parameters
    ----------
    f0 : ndarray
        F0 in Hz
    p_orig : ndarray
        Original lag as computed by YIN
    val_orig : ndarray
        Original CMNDF values as computed by YIN
    Fs : float
        Sampling frequency
    tol : float
        Tolerance for refinements in cents

    Returns
    -------
    f0_refined : ndarray
        Refined F0-trajectory
    """
    f0_refined = np.zeros_like(f0)
    voiced_idxs = np.where(f0 > 0)[0]

    f_orig = Fs / p_orig

    # find closest original YIN estimate, maximally allowed absolute deviation: R (quantization error)
    for m in voiced_idxs:
        diff_cents = np.abs(1200 * np.log2(f_orig[:, m] / f0[m]))
        candidate_idxs = np.where(diff_cents < tol)[0]

        if not candidate_idxs.size:
            f0_refined[m] = f0[m]
        else:
            f0_refined[m] = f_orig[candidate_idxs[np.argmin(val_orig[candidate_idxs, m])], m]

    return f0_refined


@njit
def probabilistic_thresholding(cmndf, thresholds, p_min, p_max, absolute_min_prob, F_axis, Fs, beta_distr,
                               parabolic_interp=True):
    """
    Probabilistic thresholding of the YIN CMNDF.

    Parameters
    ----------
    cmndf : ndarray
        Cumulative Mean Normalized Difference Function
    thresholds : ndarray
        Array of thresholds for CMNDF
    p_min : float
        Period corresponding to the lower frequency bound
    p_max : float
        Period corresponding to the upper frequency bound
    absolute_min_prob : float
        Probability to chose absolute minimum
    F_axis : ndarray
        Frequency axis
    Fs : float
        Sampling rate
    beta_distr : ndarray
        Beta distribution that defines mapping between thresholds and probabilities
    parabolic_interp : bool
        Switch to activate/deactivate parabolic interpolation

    Returns
    -------
    O_m : ndarray
        Observations for given frame
    lag_thr : ndarray
        Computed lags for every threshold
    val_thr : ndarray
        CMNDF values for computed lag
    """
    # restrict search range to interval [p_min:p_max]
    cmndf[:p_min] = np.inf
    cmndf[p_max:] = np.inf

    # find local minima (assuming that cmndf is real in [p_min:p_max], you will always find a minimum,
    # at least p_min or p_max)
    min_idxs = (np.argwhere((cmndf[1:-1] < cmndf[0:-2]) & (cmndf[1:-1] < cmndf[2:]))).flatten().astype(np.int64) + 1

    O_m = np.zeros(2 * len(F_axis))

    # return if no minima are found, e.g., when frame is silence
    if min_idxs.size == 0:
        return O_m, np.ones_like(thresholds)*p_min, np.ones_like(thresholds)

    # Optional: Parabolic Interpolation of local minima
    if parabolic_interp:
        # do not interpolate at the boarders, Numba compatible workaround for np.delete()
        min_idxs_interp = delete_numba(min_idxs, np.argwhere(min_idxs == p_min))
        min_idxs_interp = delete_numba(min_idxs_interp, np.argwhere(min_idxs_interp == p_max - 1))
        p_corr, cmndf[min_idxs_interp] = parabolic_interpolation(cmndf[min_idxs_interp - 1],
                                                                 cmndf[min_idxs_interp],
                                                                 cmndf[min_idxs_interp + 1])
    else:
        p_corr = np.zeros_like(min_idxs).astype(np.float64)

    # set p_corr=0 at the boarders (no correction done later)
    if min_idxs[0] == p_min:
        p_corr = np.concatenate((np.array([0.0]), p_corr))

    if min_idxs[-1] == p_max - 1:
        p_corr = np.concatenate((p_corr, np.array([0.0])))

    lag_thr = np.zeros_like(thresholds)
    val_thr = np.zeros_like(thresholds)

    # loop over all thresholds
    for i, threshold in enumerate(thresholds):
        # minima below absolute threshold
        min_idxs_thr = min_idxs[cmndf[min_idxs] < threshold]

        # find first local minimum
        if not min_idxs_thr.size:
            lag = np.argmin(cmndf)  # choose absolute minimum when no local minimum is found
            am_prob = absolute_min_prob
            val = np.min(cmndf)
        else:
            am_prob = 1
            lag = np.min(min_idxs_thr)  # choose first local minimum
            val = cmndf[lag]

            # correct lag
            if parabolic_interp:
                lag += p_corr[np.argmin(min_idxs_thr)]

        # ensure that lag is in [p_min:p_max]
        if lag < p_min:
            lag = p_min
        elif lag >= p_max:
            lag = p_max - 1

        lag_thr[i] = lag
        val_thr[i] = val

        idx = np.argmin(np.abs(1200 * np.log2(F_axis / (Fs / lag))))  # quantize estimated period
        O_m[idx] += am_prob * beta_distr[i]  # pYIN-Paper, Formula 4/5

    return O_m, lag_thr, val_thr


@njit
def yin_multi_thr(x, Fs, N, H, F_min, F_max, thresholds, beta_distr, absolute_min_prob, F_axis, voicing_prob,
                  parabolic_interp=True):
    """
    Applies YIN multiple times on input audio signals using different thresholds for CMNDF.

    Parameters
    ----------
    x : ndarray
        Input audio signal
    Fs : int
        Sampling rate
    N : int
        Window size
    H : int
        Hop size
    F_min : float
        Lower frequency bound
    F_max : float
        Upper frequency bound
    thresholds : ndarray
        Array of thresholds
    beta_distr : ndarray
        Beta distribution that defines mapping between thresholds and probabilities
    absolute_min_prob :float
        Probability to chose absolute minimum
    F_axis : ndarray
        Frequency axis
    voicing_prob : float
        Probability of a frame being voiced
    parabolic_interp : bool
        Switch to activate/deactivate parabolic interpolation

    Returns
    -------
    O : ndarray
        Observations based on YIN output
    rms : ndarray
        Root mean square power
    p_orig : ndarray
        Original YIN period estimates
    val_orig : ndarray
        CMNDF values corresponding to original YIN period estimates
    """

    M = int(np.floor((len(x) - N) / H)) + 1  # Compute number of estimates that will be generated
    B = len(F_axis)

    p_min = max(int(np.ceil(Fs / F_max)), 1)  # period of maximal frequency in frames
    p_max = int(np.ceil(Fs / F_min))  # period of minimal frequency in frames

    if p_max > N:
        raise Exception("The condition (Fmin >= Fs/N) was not met.")

    rms = np.zeros(M)  # RMS Power
    O = np.zeros((2 * B, M))  # every voiced state has an unvoiced state (important for later HMM modeling)
    p_orig = np.zeros((len(thresholds), M))
    val_orig = np.zeros((len(thresholds), M))

    for m in range(M):
        # Take a frame from input signal
        frame = x[m * H:m * H + N]

        # Cumulative Mean Normalized Difference Function
        cmndf = cumulative_mean_normalized_difference_function(frame, p_max)

        # compute RMS power
        rms[m] = np.sqrt(np.mean(frame ** 2))
        
        # Probabilistic Thresholding with different thresholds
        O_m, p_est_thr, val_thr = probabilistic_thresholding(cmndf, thresholds, p_min, p_max, absolute_min_prob, F_axis,
                                                             Fs, beta_distr, parabolic_interp=parabolic_interp)

        O[:, m] = O_m
        p_orig[:, m] = p_est_thr  # store original YIN estimates for later refinement
        val_orig[:, m] = val_thr  # store original cmndf value of minimum corresponding to p_est

    # normalization (pYIN-Paper, Formula 6)
    O[0:B, :] *= voicing_prob
    O[B:2 * B, :] = (1 - voicing_prob) * (1 - np.sum(O[0:B, :], axis=0)) / B
    
    return O, rms, p_orig, val_orig


@njit
def compute_transition_matrix(M, triang_distr):
    """
    Compute a transition matrix for PYIN Viterbi.

    Parameters
    ----------
    M : int
        Matrix dimension
    triang_distr : ndarray
        (Triangular) distribution, defining tolerance for jumps deviating from the main diagonal

    Returns
    -------
    A : ndarray
        Transition matrix
    """
    prob_self = 0.99
        
    A = np.zeros((2*M, 2*M))
    max_step = len(triang_distr) // 2

    for i in range(M):
        if i < max_step:
            A[i, 0:i+max_step] = prob_self * triang_distr[max_step - i:-1] / np.sum(triang_distr[max_step - i:-1])
            A[i+M, M:i+M+max_step] = prob_self * triang_distr[max_step - i:-1] / np.sum(triang_distr[max_step - i:-1])

        if i >= max_step and i < M-max_step:
            A[i, i-max_step:i+max_step+1] = prob_self * triang_distr
            A[i+M, (i+M)-max_step:(i+M)+max_step+1] = prob_self * triang_distr

        if i >= M-max_step:
            A[i, i-max_step:M] = prob_self * triang_distr[0:max_step - (i-M)] / np.sum(triang_distr[0:max_step - (i-M)])
            A[i+M, i+M-max_step:2*M] = prob_self * triang_distr[0:max_step - (i - M)] / \
                                       np.sum(triang_distr[0:max_step - (i - M)])

        A[i, i+M] = 1 - prob_self
        A[i+M, i] = 1 - prob_self
    
    return A


@njit
def viterbi_pyin(A, C, O):
    """Viterbi algorithm (pYIN variant)

        Args:
            A : ndarray
                State transition probability matrix of dimension I x I
            C : ndarray
                Initial state distribution  of dimension I X 1
            O : ndarray
                Likelihood matrix of dimension I x N

        Returns:
            idxs : ndarray
                Optimal state sequence of length N
        """
    B = O.shape[0] // 2
    M = O.shape[1]
    D = np.zeros((B * 2, M))
    E = np.zeros((B * 2, M - 1))

    idxs = np.zeros(M)

    for i in range(B * 2):
        D[i, 0] = C[i, 0] * O[i, 0]  # D matrix Intial state setting

    D[:, 0] = D[:, 0] / np.sum(D[:, 0])  # Normalization (using pYIN source code as a basis)

    for n in range(1, M):
        for i in range(B * 2):
            abyd = np.multiply(A[:, i], D[:, n-1])
            D[i, n] = np.max(abyd) * O[i, n]
            E[i, n-1] = np.argmax(abyd)

        D[:, n] = D[:, n] / np.sum(D[:, n])  # Row normalization to avoid underflow (pYIN source code sparseHMM)

    idxs[M - 1] = np.argmax(D[:, M - 1])

    for n in range(M - 2, 0, -1):
        bkd = int(idxs[n+1])  # Intermediate variable to be compatible with Numba
        idxs[n] = E[bkd, n]
    
    return idxs.astype(np.int32)


@njit
def viterbi_log_likelihood(A, C, B_O):
    """Viterbi algorithm (log variant) for solving the uncovering problem

    Notebook: C5/C5S3_Viterbi.ipynb

    Args:
        A : ndarray
            State transition probability matrix of dimension I x I
        C : ndarray
            Initial state distribution  of dimension I
        B_O : ndarray
            Likelihood matrix of dimension I x N

    Returns:
        S_opt : ndarray
            Optimal state sequence of length N
    """
    I = A.shape[0]    # Number of states
    N = B_O.shape[1]  # Length of observation sequence
    tiny = np.finfo(0.).tiny
    A_log = np.log(A + tiny)
    C_log = np.log(C + tiny)
    B_O_log = np.log(B_O + tiny)

    # Initialize D and E matrices
    D_log = np.zeros((I, N))
    E = np.zeros((I, N-1)).astype(np.int32)
    D_log[:, 0] = C_log + B_O_log[:, 0]

    # Compute D and E in a nested loop
    for n in range(1, N):
        for i in range(I):
            temp_sum = A_log[:, i] + D_log[:, n-1]
            D_log[i, n] = np.max(temp_sum) + B_O_log[i, n]
            E[i, n-1] = np.argmax(temp_sum)

    # Backtracking
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D_log[:, -1])
    for n in range(N-2, -1, -1):
        S_opt[n] = E[int(S_opt[n+1]), n]

    return S_opt


@njit
def delete_numba(arr, num):
    """Delete number from array, Numba compatible. Inspired by:
        https://stackoverflow.com/questions/53602663/delete-a-row-in-numpy-array-in-numba
    """
    mask = np.zeros(len(arr), dtype=np.int64) == 0
    mask[np.where(arr == num)[0]] = False
    return arr[mask]
