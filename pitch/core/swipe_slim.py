"""
| Description: libf0 SWIPE slim implementation
| Contributors: Sebastian Rosenzweig, Simon Schwär, Meinard Müller
| License: The MIT license, https://opensource.org/licenses/MIT
| This file is part of libf0.
"""
import numpy as np
import librosa
from .yin import parabolic_interpolation
from scipy.interpolate import interp1d


def swipe_slim(x, Fs=22050, H=256, F_min=55.0, F_max=1760.0, R=10, strength_threshold=0):
    """
    Slim and didactical implementation of a sawtooth waveform inspired pitch estimator (SWIPE).
    This version uses a log-frequency spectrogram instead of ERB filters. Furthermore, it is implemented more
    efficiently. See `swipe()` for the original implementation.

    .. [#] A. Camacho and J. G. Harris,
       "A sawtooth waveform inspired pitch estimator for speech and music."
       The Journal of the Acoustical Society of America, vol. 124, no. 3, pp. 1638–1652, Sep. 2008

    Parameters
    ----------
    x : ndarray
        Audio signal
    Fs : int
        Sampling rate
    H : int
        Hop size
    F_min : float or int
        Minimal frequency
    F_max : float or int
        Maximal frequency
    R : float
        resolution of the pitch candidate bins in cents (default = 10)
    strength_threshold : float
        confidence threshold [0, 1] for the pitch detection (default value = 0)

    Returns
    -------
    f0 : ndarray
        Estimated F0-trajectory
    t : ndarray
        Time axis
    conf : ndarray
        Confidence / Pitch Strength
    """

    # compute time and frequency axis
    t = np.arange(0, len(x), H) / Fs  # time axis
    F_coef_log = np.arange(0, np.log2(Fs/2/F_min), R/1200)
    F_coef_log_hz = F_min * 2 ** F_coef_log  # pitch candidates

    # pre-compute kernels, one kernel for each pitch candidate in range [F_min : F_max]
    F_min_idx = np.argmin(np.abs(F_coef_log_hz - F_min))
    F_max_idx = np.argmin(np.abs(F_coef_log_hz - F_max))
    B = F_max_idx - F_min_idx  # Number of pitch candidates
    kernels = np.zeros((B, len(F_coef_log_hz)))
    for i, f in enumerate(F_coef_log_hz[F_min_idx:F_max_idx]):
        kernels[i, :] = compute_kernel(f, F_coef_log_hz)

    # determine optimal window length for each candidate
    L_opt = np.log2(Fs * 8 / np.array([F_min, F_max]))  # exponents for optimal window sizes 2^L, see paper Section II.G
    L_rnd = np.arange(np.round(L_opt[1]), np.round(L_opt[0])+1).astype(np.int32)  # range of rounded exponents
    N_pow2 = 2 ** L_rnd  # Compute rounded power-2 windows sizes
    # Quantization error between optimal window size (see paper Section II.G) and rounded power-2 windows size
    # Using only the largest N here, since errors for other N can be derived from err by subtracting exponent (cyclic)
    err = np.abs(np.log2(8 * Fs / F_coef_log_hz[F_min_idx:F_max_idx]) - np.log2(np.max(N_pow2)))

    S = np.zeros((B, len(t)))  # "pitch-strength" matrix

    # loop through all window sizes
    for octave, N in enumerate(N_pow2):
        # Compute STFT
        x_pad = np.pad(x, (0, N))  # to avoid problems during time axis interpolation
        H = N // 2
        X = librosa.stft(x_pad, n_fft=N, hop_length=H, win_length=N, window='hann', pad_mode='constant', center=True)
        Y = np.abs(X)
        T_coef_lin_s = np.arange(0, X.shape[1]) * H / Fs
        F_coef_lin_hz = np.arange(N // 2 + 1) * Fs / N

        # Resample to log-frequency axis
        compute_Y_log = interp1d(F_coef_lin_hz, Y, kind='cubic', axis=0)
        Y_log = compute_Y_log(F_coef_log_hz)

        # Normalize magnitudes
        Y_log /= np.sqrt(np.sum(Y_log ** 2, axis=0)) + np.finfo(float).eps

        # Correlate kernels with log-spectrum for pitch candidates where N is optimal
        S_N = np.matmul(kernels, Y_log)

        # Resample time axis
        compute_S_N_res = interp1d(T_coef_lin_s, S_N, kind='linear', axis=1)
        S_N_res = compute_S_N_res(t)

        # Weight pitch strength according to quantization error
        candidates = (err > octave - 1) & (err < octave + 1)  # consider pitches +/- 1 octave from current window
        mu = 1 - np.abs(err[candidates] - octave)

        S[candidates, :] += np.multiply(mu.reshape(-1, 1), S_N_res[candidates, :])

    # Obtain pitch estimates and corresponding confidence
    max_indices = np.argmax(S, axis=0)
    conf = np.max(S, axis=0)

    # Parabolic Interpolation of pitch estimates for refinement
    time_idx = np.arange(S.shape[1])
    indeces_shift, _ = parabolic_interpolation(S[max_indices-1, time_idx],
                                               S[max_indices, time_idx],
                                               S[max_indices+1, time_idx])
    compute_f0_log = interp1d(np.arange(len(F_coef_log)), F_coef_log, kind='linear')
    f0_hz = F_min * 2 ** compute_f0_log(max_indices+indeces_shift)

    # Thresholding
    f0_hz[conf < strength_threshold] = 0  # discard estimates where confidence is low

    return f0_hz, t, conf


def compute_kernel(f, F_coef_log_hz):
    """
    Compute a SWIPE' kernel.

    Parameters
    ----------
    f : float
        Frequency in Hz
    F_coef_log_hz :
        Logarithmic frequency axis in Hz

    Returns
    -------
    k : ndarray
        Kernel
    """
    k = np.zeros(len(F_coef_log_hz))
    n_harmonics = np.floor(F_coef_log_hz[-1] / f).astype(np.int32)
    prime_numbers = prime_and_one(100)[:n_harmonics]  # only consider prime harmonics for kernel peaks

    ratio = F_coef_log_hz / f

    # loop through all prime harmonics
    for p in prime_numbers:
        a = np.abs(ratio - p)  # normalized distance between harmonic and current pitch candidate
        main_peak_bins = a < 0.25
        k[main_peak_bins] = np.cos(np.dot(np.array(2 * np.pi).reshape(-1, 1),
                                          ratio[main_peak_bins].reshape(1, -1))).flatten()
        valley_bins = np.logical_and(0.25 < a, a < 0.75)
        k[valley_bins] += np.cos(np.dot(np.array(2 * np.pi).reshape(-1, 1),
                                        ratio[valley_bins].reshape(1, -1))).flatten() / 2

    # Apply decay
    k = np.multiply(k, np.sqrt(1.0 / F_coef_log_hz))

    # K+-normalize kernel
    k = k / np.linalg.norm(k[k > 0])

    return k


def prime_and_one(upto=1000000):
    """
    Returns a set of prime numbers, adapted from http://rebrained.com/?p=458

    Parameters
    ----------
    upto : int
        Find prime numbers up to this number

    Returns
    -------
    A set of prime numbers including 1 & 2
    """
    primes = np.arange(3, upto+1, 2)
    isprime = np.ones((upto-1)//2, dtype=np.bool8)
    for factor in primes[:int(np.sqrt(upto))//2]:
        if isprime[(factor-2)//2]:
            isprime[(factor*3-2)//2::factor] = 0
    return np.concatenate((np.array([1, 2]), primes[isprime]))
