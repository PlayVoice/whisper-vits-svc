# %%
import numpy as np
import librosa
import librosa.display

from scipy.signal import medfilt
from matplotlib import pyplot as plt
from .featureExtraction import read_audio
from .utils import *


# %%
def calc_tempo(path_audio):
    """ Calculate audio tempo
    ----------
    Parameters:
        path_audio: str
    
    ----------
    Returns: 
        tempo: float
    
    """
    target_sr = 22050
    y, _ = read_audio(path_audio, sr=target_sr)
    onset_strength = librosa.onset.onset_strength(y, sr=target_sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_strength, sr=target_sr)
    return tempo


def one_beat_frame_size(tempo):
    """ Calculate frame size of 1 beat
    ----------
    Parameters:
        tempo: float
    
    ----------
    Returns: 
        tempo: int
    
    """
    return np.int(np.round(60 / tempo * 100))


def median_filter_pitch(pitch, medfilt_size, weight):
    """ Smoothing pitch using median filter
    ----------
    Parameters:
        pitch: array
        medfilt_size: int
        weight: float
    
    ----------
    Returns: 
        pitch: array
    
    """

    medfilt_size = np.int(medfilt_size * weight)
    if medfilt_size % 2 == 0:
        medfilt_size += 1
    return np.round(medfilt(pitch, medfilt_size))


def clean_note_frames(note, min_note_len=5):
    """ Remove short pitch frames 
    ----------
    Parameters:
        note: array
        min_note_len: int
        
    ----------
    Returns: 
        output: array
    
    """

    prev_pitch = 0
    prev_pitch_start = 0
    output = np.copy(note)
    for i in range(len(note)):
        pitch = note[i]
        if pitch != prev_pitch:
            prev_pitch_duration = i - prev_pitch_start
            if prev_pitch_duration < min_note_len:
                output[prev_pitch_start:i] = [0] * prev_pitch_duration
            prev_pitch = pitch
            prev_pitch_start = i
    return output


def makeSegments(note):
    """ Make segments of notes
    ----------
    Parameters:
        note: array
               
    ----------
    Returns: 
        startSeg: starting points (array)
        endSeg: ending points (array)
    
    """
    startSeg = []
    endSeg = []
    flag = -1
    if note[0] > 0:
        startSeg.append(0)
        flag *= -1
    for i in range(0, len(note) - 1):
        if note[i] != note[i + 1]:
            if flag < 0:
                startSeg.append(i + 1)
                flag *= -1
            else:
                if note[i + 1] == 0:
                    endSeg.append(i)
                    flag *= -1
                else:
                    endSeg.append(i)
                    startSeg.append(i + 1)
    return startSeg, endSeg


def remove_short_segment(idx, note_cleaned, start, end, minLength):
    """ Remove short segments
    ----------
    Parameters:
        idx: (int)
        note_cleaned: (array)
        start: starting points (array)
        end: ending points (array)
        minLength: (int)
               
    ----------
    Returns: 
        note_cleaned: (array)
            
    """

    len_seg = end[idx] - start[idx]
    if len_seg < minLength:
        if (start[idx + 1] - end[idx] > minLength) and (start[idx] - end[idx - 1] > minLength):
            note_cleaned[start[idx] : end[idx] + 1] = [0] * (len_seg + 1)
    return note_cleaned


def remove_octave_error(idx, note_cleaned, start, end):
    """ Remove octave error
    ----------
    Parameters:
        idx: (int)
        note_cleaned: (array)
        start: starting points (array)
        end: ending points (array)
               
    ----------
    Returns: 
        note_cleaned: (array)
            
    """
    len_seg = end[idx] - start[idx]
    if (note_cleaned[start[idx - 1]] == note_cleaned[start[idx + 1]]) and (
        note_cleaned[start[idx]] != note_cleaned[start[idx + 1]]
    ):
        if np.abs(note_cleaned[start[idx]] - note_cleaned[start[idx + 1]]) % 12 == 0:
            note_cleaned[start[idx] - 1 : end[idx] + 1] = [note_cleaned[start[idx + 1]]] * (
                len_seg + 2
            )
    return note_cleaned


def clean_segment(note, minLength):
    """ clean note segments
    ----------
    Parameters:
        note: (array)
        minLength: (int)
               
    ----------
    Returns: 
        note_cleaned: (array)
            
    """

    note_cleaned = np.copy(note)
    start, end = makeSegments(note_cleaned)

    for i in range(1, len(start) - 1):
        note_cleaned = remove_short_segment(i, note_cleaned, start, end, minLength)
        note_cleaned = remove_octave_error(i, note_cleaned, start, end)
    return note_cleaned


def refine_note(est_note, tempo):
    """ main: refine note segments
    ----------
    Parameters:
        est_note: (array)
        tempo: (float)
               
    ----------
    Returns: 
        est_pitch_mf3_v: (array)
            
    """
    one_beat_size = one_beat_frame_size(tempo)
    est_note_mf1 = median_filter_pitch(est_note, one_beat_size, 1 / 8)
    est_note_mf2 = median_filter_pitch(est_note_mf1, one_beat_size, 1 / 4)
    est_note_mf3 = median_filter_pitch(est_note_mf2, one_beat_size, 1 / 3)

    vocing = est_note_mf1 > 0
    est_pitch_mf3_v = vocing * est_note_mf3
    est_pitch_mf3_v = clean_note_frames(est_pitch_mf3_v, int(one_beat_size * 1 / 8))
    est_pitch_mf3_v = clean_segment(est_pitch_mf3_v, int(one_beat_size * 1 / 4))
    return est_pitch_mf3_v

