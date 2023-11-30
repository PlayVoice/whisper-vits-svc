"""
| Description: libf0 utility functions
| Contributors: Sebastian Rosenzweig, Simon Schwär, Meinard Müller
| License: The MIT license, https://opensource.org/licenses/MIT
| This file is part of libf0.
"""
import numpy as np


def sonify_trajectory_with_sinusoid(f0, t, audio_len, confidence=None, Fs=22050, smooth_len=11):
    """
    Sonification of trajectory with sinusoidal. Adapted from FMP notebook: C8/C8S2_FundFreqTracking.ipynb

    Parameters
    ----------
    f0 : ndarray
        F0-trajectory
    t : ndarray
        Time axis
    audio_len : int
        Desired audio length in samples
    confidence : None or ndarray
        Confidence values for amplitude control
    Fs : int
        Sampling rate
    smooth_len : int
        Smoothing filter length to avoid clicks in the sonification

    Returns
    -------
    x_soni : ndarray
        Sonified F0-trajectory
    """
    if confidence is None:
        confidence = np.ones_like(f0)

    # initialize
    x_soni = np.zeros(audio_len)
    amplitude_mod = np.zeros(audio_len)

    # Computation of hop size
    sine_len = int(t[1] * Fs)

    t = np.arange(0, sine_len) / Fs
    phase = 0

    # loop over all F0 values, ensure continuous phase
    for idx in np.arange(0, len(f0)):
        cur_f = f0[idx]
        cur_amp = confidence[idx]

        if cur_f == 0:
            phase = 0
            continue

        cur_soni = np.sin(2*np.pi*(cur_f*t+phase))
        diff = np.maximum(0, (idx+1)*sine_len - len(x_soni))
        if diff > 0:
            x_soni[idx * sine_len:(idx + 1) * sine_len - diff] = cur_soni[:-diff]
            amplitude_mod[idx * sine_len:(idx + 1) * sine_len - diff] = cur_amp
        else:
            x_soni[idx*sine_len:(idx+1)*sine_len-diff] = cur_soni
            amplitude_mod[idx*sine_len:(idx+1)*sine_len-diff] = cur_amp

        phase += cur_f * sine_len / Fs
        phase -= 2 * np.round(phase/2)

    # filter amplitudes to avoid transients
    amplitude_mod = np.convolve(amplitude_mod, np.hanning(smooth_len)/np.sum(np.hanning(smooth_len)), 'same')
    x_soni = x_soni * amplitude_mod
    return x_soni


def hz_to_cents(F, F_ref=55.0):
    """
    Converts frequency in Hz to cents.

    Parameters
    ----------
    F : float or ndarray
        Frequency value in Hz
    F_ref : float
        Reference frequency in Hz (Default value = 55.0)
    Returns
    -------
    F_cents : float or ndarray
        Frequency in cents
    """

    # Avoid division by 0
    F_temp = np.array(F).astype(float)
    F_temp[F_temp == 0] = np.nan

    F_cents = 1200 * np.log2(F_temp / F_ref)

    return F_cents


def cents_to_hz(F_cents, F_ref=55.0):
    """
    Converts frequency in cents to Hz.

    Parameters
    ----------
    F_cents : float or ndarray
        Frequency in cents
    F_ref : float
        Reference frequency in Hz (Default value = 55.0)
    Returns
    -------
    F : float or ndarray
        Frequency in Hz
    """
    F = F_ref * 2 ** (F_cents / 1200)

    # Avoid NaN output
    F = np.nan_to_num(F, copy=False, nan=0)

    return F
