# -*- coding: utf-8 -*-
import librosa
from pydub import AudioSegment
import pathlib

# from pydub.playback import play
import numpy as np
import os

PATH_PROJECT = os.path.dirname(os.path.realpath(__file__))


def read_audio(filepath, sr=None):
    path = pathlib.Path(filepath)
    extenstion = path.suffix.replace(".", "")
    if extenstion == "mp3":
        sound = AudioSegment.from_mp3(filepath)
    else:
        sound = AudioSegment.from_file(filepath)
    # sound = sound[start * 1000 : end * 1000]
    sound = sound.set_channels(1)
    if sr == None:
        sr = sound.frame_rate
    sound = sound.set_frame_rate(sr)
    samples = sound.get_array_of_samples()
    y = np.array(samples).T.astype(np.float32)

    return y, sr


def spec_extraction(file_name, win_size):

    y, _ = read_audio(file_name, sr=8000)

    S = librosa.core.stft(y, n_fft=1024, hop_length=80, win_length=1024)
    x_spec = np.abs(S)
    x_spec = librosa.core.power_to_db(x_spec, ref=np.max)
    x_spec = x_spec.astype(np.float32)
    num_frames = x_spec.shape[1]

    # for padding
    padNum = num_frames % win_size
    if padNum != 0:
        len_pad = win_size - padNum
        padding_feature = np.zeros(shape=(513, len_pad))
        x_spec = np.concatenate((x_spec, padding_feature), axis=1)
        num_frames = num_frames + len_pad

    x_test = []
    for j in range(0, num_frames, win_size):
        x_test_tmp = x_spec[:, range(j, j + win_size)].T
        x_test.append(x_test_tmp)
    x_test = np.array(x_test)

    # for standardization
    path_project = pathlib.Path(__file__).parent.parent
    x_train_mean = np.load(f"{path_project}/data/x_train_mean.npy")
    x_train_std = np.load(f"{path_project}/data/x_train_std.npy")
    x_test = (x_test - x_train_mean) / (x_train_std + 0.0001)
    x_test = x_test[:, :, :, np.newaxis]
    return x_test, x_spec
