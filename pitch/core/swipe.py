"""
| Description: libf0 SWIPE implementation
| Contributors: Sebastian Rosenzweig, Vojtěch Pešek, Simon Schwär, Meinard Müller
| License: The MIT license, https://opensource.org/licenses/MIT
| This file is part of libf0.
"""
from scipy import interpolate
import numpy as np
import librosa


def swipe(x, Fs=22050, H=256, F_min=55.0, F_max=1760.0, dlog2p=1 / 96, derbs=0.1, strength_threshold=0):
    """
    Implementation of a sawtooth waveform inspired pitch estimator (SWIPE).
    This version of the algorithm follows the original implementation, see `swipe_slim` for a more efficient
    alternative.

    .. [#] Arturo Camacho and John G. Harris,
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
    dlog2p : float
        resolution of the pitch candidate bins in octaves (default value = 1/96 -> 96 bins per octave)
    derbs : float
        resolution of the ERB bands (default value = 0.1)
    strength_threshold : float
        confidence threshold [0, 1] for the pitch detection (default value = 0)

    Returns
    -------
    f0 : ndarray
        Estimated F0-trajectory
    t : ndarray
        Time axis
    strength : ndarray
        Confidence/Pitch Strength
    """

    t = np.arange(0, len(x), H) / Fs  # Times

    # Compute pitch candidates
    pc = 2 ** np.arange(np.log2(F_min), np.log2(F_max), dlog2p)

    # Pitch strength matrix
    S = np.zeros((len(pc), len(t)))

    # Determine P2-WSs [max, min]
    log_ws_max = np.ceil(np.log2((8 / F_min) * Fs))
    log_ws_min = np.floor(np.log2((8 / F_max) * Fs))

    # P2-WSs - window sizes in samples
    ws = 2 ** np.arange(log_ws_max, log_ws_min - 1, -1, dtype=np.int32)
    # print(f'window sizes in samples: {ws}')

    # Determine window sizes used by each pitch candidate
    log2pc = np.arange(np.log2(F_min), np.log2(F_max), dlog2p)
    d = log2pc - np.log2(np.divide(8 * Fs, ws[0]))

    # Create ERBs spaced frequencies (in Hertz)
    fERBs = erbs2hz(np.arange(hz2erbs(pc[0] / 4), hz2erbs(Fs / 2), derbs))

    for i in range(0, len(ws)):
        N = ws[i]
        H = int(N / 2)

        x_zero_padded = np.concatenate([x, np.zeros(N)])

        X = librosa.stft(x_zero_padded, n_fft=N, hop_length=H, pad_mode='constant', center=True)
        ti = librosa.frames_to_time(np.arange(0, X.shape[1]), sr=Fs, hop_length=H, n_fft=N)
        f = librosa.fft_frequencies(sr=Fs, n_fft=N)

        ti = np.insert(ti, 0, 0)
        ti = np.delete(ti, -1)

        spectrum = np.abs(X)
        magnitude = resample_ferbs(spectrum, f, fERBs)
        loudness = np.sqrt(magnitude)

        # Select candidates that use this window size
        # First window
        if i == 0:
            j = np.argwhere(d < 1).flatten()
            k = np.argwhere(d[j] > 0).flatten()
        # Last Window
        elif i == len(ws) - 1:
            j = np.argwhere(d - i > -1).flatten()
            k = np.argwhere(d[j] - i < 0).flatten()
        else:
            j = np.argwhere(np.abs(d - i) < 1).flatten()
            k = np.arange(0, len(j))

        pc_to_compute = pc[j]

        pitch_strength = pitch_strength_all_candidates(fERBs, loudness, pc_to_compute)

        resampled_pitch_strength = resample_time(pitch_strength, t, ti)

        lambda_ = d[j[k]] - i
        mu = np.ones(len(j))
        mu[k] = 1 - np.abs(lambda_)

        S[j, :] = S[j, :] + np.multiply(
            np.ones(resampled_pitch_strength.shape) * mu.reshape((mu.shape[0], 1)),
            resampled_pitch_strength
        )

    # Fine-tune the pitch using parabolic interpolation
    pitches, strength = parabolic_int(S, strength_threshold, pc)

    pitches[np.where(np.isnan(pitches))] = 0  # avoid NaN output

    return pitches, t, strength


def nyquist(Fs):
    """Nyquist Frequency"""
    return Fs / 2


def F_coef(k, N, Fs):
    """Physical frequency of STFT coefficients"""
    return (k * Fs) / N


def T_coef(m, H, Fs):
    """Physical time of STFT coefficients"""
    return m * H / Fs


def stft_with_f_t(y, N, H, Fs):
    """STFT wrapper"""
    x = librosa.stft(y, int(N), int(H), pad_mode='constant', center=True)
    f = F_coef(np.arange(0, x.shape[0]), N, Fs)
    t = T_coef(np.arange(0, x.shape[1]), H, Fs)

    return x, f, t


def hz2erbs(hz):
    """Convert Hz to ERB scale"""
    return 21.4 * np.log10(1 + hz / 229)


def erbs2hz(erbs):
    """Convert ERB to Hz"""
    return (10 ** np.divide(erbs, 21.4) - 1) * 229


def pitch_strength_all_candidates(ferbs, loudness, pitch_candidates):
    """Compute pitch strength for all pitch candidates"""
    # Normalize loudness
    normalization_loudness = np.full_like(loudness, np.sqrt(np.sum(loudness * loudness, axis=0)))
    with np.errstate(divide='ignore', invalid='ignore'):
        loudness = loudness / normalization_loudness

    # Create pitch salience matrix
    S = np.zeros((len(pitch_candidates), loudness.shape[1]))

    for j in range(0, len(pitch_candidates)):
        S[j, :] = pitch_strength_one(ferbs, loudness, pitch_candidates[j])
    return S


def pitch_strength_one(erbs_frequencies, normalized_loudness, pitch_candidate):
    """Compute pitch strength for one pitch candidate"""
    number_of_harmonics = np.floor(erbs_frequencies[-1] / pitch_candidate - 0.75).astype(np.int32)
    k = np.zeros(erbs_frequencies.shape)

    # f_prime / f
    q = erbs_frequencies / pitch_candidate

    for i in np.concatenate(([1], primes(number_of_harmonics))):
        a = np.abs(q - i)
        p = a < 0.25
        k[p] = np.cos(np.dot(2 * np.pi, q[p]))
        v = np.logical_and(0.25 < a, a < 0.75)
        k[v] = k[v] + np.cos(np.dot(2 * np.pi, q[v])) / 2

    # Apply envelope
    k = np.multiply(k, np.sqrt(1.0 / erbs_frequencies))

    # K+-normalize kernel
    k = k / np.linalg.norm(k[k > 0])

    # Compute pitch strength
    S = np.dot(k, normalized_loudness)
    return S


def resample_ferbs(spectrum, f, ferbs):
    """Resample to ERB scale"""
    magnitude = np.zeros((len(ferbs), spectrum.shape[1]))

    for t in range(spectrum.shape[1]):
        spl = interpolate.splrep(f, spectrum[:, t])
        interpolate.splev(ferbs, spl)

        magnitude[:, t] = interpolate.splev(ferbs, spl)

    return np.maximum(magnitude, 0)


def resample_time(pitch_strength, resampled_time, ti):
    """Resample time axis"""
    if pitch_strength.shape[1] > 0:
        pitch_strength = interpolate_one_candidate(pitch_strength, ti, resampled_time)
    else:
        pitch_strength = np.kron(np.ones((len(pitch_strength), len(resampled_time))), np.NaN)
    return pitch_strength


def interpolate_one_candidate(pitch_strength, ti, resampled_time):
    """Interpolate time axis"""
    pitch_strength_interpolated = np.zeros((pitch_strength.shape[0], len(resampled_time)))

    for s in range(pitch_strength.shape[0]):
        t_i = interpolate.interp1d(ti, pitch_strength[s, :], 'linear', bounds_error=True)
        pitch_strength_interpolated[s, :] = t_i(resampled_time)

    return pitch_strength_interpolated


def parabolic_int(pitch_strength, strength_threshold, pc):
    """Parabolic interpolation between pitch candidates using pitch strength"""
    p = np.full((pitch_strength.shape[1],), np.NaN)
    s = np.full((pitch_strength.shape[1],), np.NaN)

    for j in range(pitch_strength.shape[1]):
        i = np.argmax(pitch_strength[:, j])
        s[j] = pitch_strength[i, j]

        if s[j] < strength_threshold:
            continue

        if i == 0:
            p[j] = pc[0]
        elif i == len(pc) - 1:
            p[j] = pc[0]
        else:
            I = np.arange(i - 1, i + 2)
            tc = 1 / pc[I]
            ntc = np.dot((tc / tc[1] - 1), 2 * np.pi)
            if np.any(np.isnan(pitch_strength[I, j])):
                s[j] = np.nan
                p[j] = np.nan
            else:
                c = np.polyfit(ntc, pitch_strength[I, j], 2)
                ftc = 1 / 2 ** np.arange(np.log2(pc[I[0]]), np.log2(pc[I[2]]), 1 / 12 / 64)
                nftc = np.dot((ftc / tc[1] - 1), 2 * np.pi)
                poly = np.polyval(c, nftc)
                k = np.argmax(poly)
                s[j] = poly[k]
                p[j] = 2 ** (np.log2(pc[I[0]]) + k / 12 / 64)
    return p, s


def primes(n):
    """Returns a set of n prime numbers"""
    small_primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
                             97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
                             191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
                             283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397,
                             401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
                             509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619,
                             631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743,
                             751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863,
                             877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997])

    b = small_primes <= n
    return small_primes[b]
