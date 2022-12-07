import time

import numpy as np
import torch
import torchaudio
from scipy.ndimage import maximum_filter1d, uniform_filter1d


def timeit(func):
    def run(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print('executing \'%s\' costed %.3fs' % (func.__name__, time.time() - t))
        return res

    return run


# @timeit
def _window_maximum(arr, win_sz):
    return maximum_filter1d(arr, size=win_sz)[win_sz // 2: win_sz // 2 + arr.shape[0] - win_sz + 1]


# @timeit
def _window_rms(arr, win_sz):
    filtered = np.sqrt(uniform_filter1d(np.power(arr, 2), win_sz) - np.power(uniform_filter1d(arr, win_sz), 2))
    return filtered[win_sz // 2: win_sz // 2 + arr.shape[0] - win_sz + 1]


def level2db(levels, eps=1e-12):
    return 20 * np.log10(np.clip(levels, a_min=eps, a_max=1))


def _apply_slice(audio, begin, end):
    if len(audio.shape) > 1:
        return audio[:, begin: end]
    else:
        return audio[begin: end]


class Slicer:
    def __init__(self,
                 sr: int,
                 db_threshold: float = -40,
                 min_length: int = 5000,
                 win_l: int = 300,
                 win_s: int = 20,
                 max_silence_kept: int = 500):
        self.db_threshold = db_threshold
        self.min_samples = round(sr * min_length / 1000)
        self.win_ln = round(sr * win_l / 1000)
        self.win_sn = round(sr * win_s / 1000)
        self.max_silence = round(sr * max_silence_kept / 1000)
        if not self.min_samples >= self.win_ln >= self.win_sn:
            raise ValueError('The following condition must be satisfied: min_length >= win_l >= win_s')
        if not self.max_silence >= self.win_sn:
            raise ValueError('The following condition must be satisfied: max_silence_kept >= win_s')

    @timeit
    def slice(self, audio):
        samples = audio
        if samples.shape[0] <= self.min_samples:
            return {"0": {"slice": False, "split_time": f"0,{len(audio)}"}}
        # get absolute amplitudes
        abs_amp = np.abs(samples - np.mean(samples))
        # calculate local maximum with large window
        win_max_db = level2db(_window_maximum(abs_amp, win_sz=self.win_ln))
        sil_tags = []
        left = right = 0
        while right < win_max_db.shape[0]:
            if win_max_db[right] < self.db_threshold:
                right += 1
            elif left == right:
                left += 1
                right += 1
            else:
                if left == 0:
                    split_loc_l = left
                else:
                    sil_left_n = min(self.max_silence, (right + self.win_ln - left) // 2)
                    rms_db_left = level2db(_window_rms(samples[left: left + sil_left_n], win_sz=self.win_sn))
                    split_win_l = left + np.argmin(rms_db_left)
                    split_loc_l = split_win_l + np.argmin(abs_amp[split_win_l: split_win_l + self.win_sn])
                if len(sil_tags) != 0 and split_loc_l - sil_tags[-1][1] < self.min_samples and right < win_max_db.shape[
                    0] - 1:
                    right += 1
                    left = right
                    continue
                if right == win_max_db.shape[0] - 1:
                    split_loc_r = right + self.win_ln
                else:
                    sil_right_n = min(self.max_silence, (right + self.win_ln - left) // 2)
                    rms_db_right = level2db(_window_rms(samples[right + self.win_ln - sil_right_n: right + self.win_ln],
                                                        win_sz=self.win_sn))
                    split_win_r = right + self.win_ln - sil_right_n + np.argmin(rms_db_right)
                    split_loc_r = split_win_r + np.argmin(abs_amp[split_win_r: split_win_r + self.win_sn])
                sil_tags.append((split_loc_l, split_loc_r))
                right += 1
                left = right
        if left != right:
            sil_left_n = min(self.max_silence, (right + self.win_ln - left) // 2)
            rms_db_left = level2db(_window_rms(samples[left: left + sil_left_n], win_sz=self.win_sn))
            split_win_l = left + np.argmin(rms_db_left)
            split_loc_l = split_win_l + np.argmin(abs_amp[split_win_l: split_win_l + self.win_sn])
            sil_tags.append((split_loc_l, samples.shape[0]))
        if len(sil_tags) == 0:
            return {"0": {"slice": False, "split_time": f"0,{len(audio)}"}}
        else:
            chunks = []
            # 第一段静音并非从头开始，补上有声片段
            if sil_tags[0][0]:
                chunks.append({"slice": False, "split_time": f"0,{sil_tags[0][0]}"})
            for i in range(0, len(sil_tags)):
                # 标识有声片段（跳过第一段）
                if i:
                    chunks.append({"slice": False, "split_time": f"{sil_tags[i - 1][1]},{sil_tags[i][0]}"})
                # 标识所有静音片段
                chunks.append({"slice": True, "split_time": f"{sil_tags[i][0]},{sil_tags[i][1]}"})
            # 最后一段静音并非结尾，补上结尾片段
            if sil_tags[-1][1] != len(audio):
                chunks.append({"slice": False, "split_time": f"{sil_tags[-1][1]},{len(audio)}"})
            chunk_dict = {}
            for i in range(len(chunks)):
                chunk_dict[str(i)] = chunks[i]
            return chunk_dict


def cut(audio_path, db_thresh=-30, min_len=5000, win_l=300, win_s=20, max_sil_kept=500):
    audio, sr = torchaudio.load(audio_path)
    if len(audio.shape) == 2 and audio.shape[1] >= 2:
        audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio = audio.cpu().numpy()[0]

    slicer = Slicer(
        sr=sr,
        db_threshold=db_thresh,
        min_length=min_len,
        win_l=win_l,
        win_s=win_s,
        max_silence_kept=max_sil_kept
    )
    chunks = slicer.slice(audio)
    return chunks


def chunks2audio(audio_path, chunks):
    chunks = dict(chunks)
    audio, sr = torchaudio.load(audio_path)
    if len(audio.shape) == 2 and audio.shape[1] >= 2:
        audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio = audio.cpu().numpy()[0]
    result = []
    for k, v in chunks.items():
        tag = v["split_time"].split(",")
        result.append((v["slice"], audio[int(tag[0]):int(tag[1])]))
    return result, sr


