import os
import numpy as np
from pydub import AudioSegment
import pathlib


def check_and_make_dir(path_dir):
    if not os.path.exists(os.path.dirname(path_dir)):
        os.makedirs(os.path.dirname(path_dir))


def get_filename_wo_extension(path_dir):
    return pathlib.Path(path_dir).stem


def note2pitch(pitch):
    """ Convert MIDI number to freq.
    ----------
    Parameters:
        pitch: MIDI note numbers of pitch (array)
    
    ----------
    Returns: 
        pitch: freqeuncy of pitch (array)
    """

    pitch = np.array(pitch)
    pitch[pitch > 0] = 2 ** ((pitch[pitch > 0] - 69) / 12.0) * 440
    return pitch


def pitch2note(pitch):
    """ Convert freq to MIDI number
    ----------
    Parameters:
        pitch: freqeuncy of pitch (array)
    
    ----------
    Returns: 
        pitch: MIDI note numbers of pitch (array)
    """
    pitch = np.array(pitch)
    pitch[pitch > 0] = np.round((69.0 + 12.0 * np.log2(pitch[pitch > 0] / 440.0)))
    return pitch


a = np.array([0, 0, 0, 1, 2, 3, 5, 0, 0, 0, 1, 2, 4, 5])
b = a[a > 0] * 2
print(b)
