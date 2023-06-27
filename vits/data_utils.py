import os
import numpy as np
import random
import torch
import torch.utils.data


from vits.utils import load_wav_to_torch


def load_filepaths(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths = [line.strip().split(split) for line in f]
    return filepaths


class TextAudioSpeakerSet(torch.utils.data.Dataset):
    def __init__(self, filename, hparams):
        self.items = load_filepaths(filename)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.segment_size = hparams.segment_size
        self.hop_length = hparams.hop_length
        self._filter()
        print(f'----------{len(self.items)}----------')

    def _filter(self):
        lengths = []
        items_new = []
        items_min = int(self.segment_size / self.hop_length * 4)  # 1 S
        items_max = int(self.segment_size / self.hop_length * 16)  # 4 S
        for wavpath, spec, pitch, vec, ppg, spk in self.items:
            if not os.path.isfile(wavpath):
                continue
            if not os.path.isfile(spec):
                continue
            if not os.path.isfile(pitch):
                continue
            if not os.path.isfile(vec):
                continue
            if not os.path.isfile(ppg):
                continue
            if not os.path.isfile(spk):
                continue
            temp = np.load(pitch)
            usel = int(temp.shape[0] - 1)  # useful length
            if (usel < items_min):
                continue
            if (usel >= items_max):
                usel = items_max
            items_new.append([wavpath, spec, pitch, vec, ppg, spk, usel])
            lengths.append(usel)
        self.items = items_new
        self.lengths = lengths

    def read_wav(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        assert sampling_rate == self.sampling_rate, f"error: this sample rate of {filename} is {sampling_rate}"
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        return audio_norm

    def __getitem__(self, index):
        return self.my_getitem(index)

    def __len__(self):
        return len(self.items)

    def my_getitem(self, idx):
        item = self.items[idx]
        # print(item)
        wav = item[0]
        spe = item[1]
        pit = item[2]
        vec = item[3]
        ppg = item[4]
        spk = item[5]
        use = item[6]

        wav = self.read_wav(wav)
        spe = torch.load(spe)

        pit = np.load(pit)
        vec = np.load(vec)
        vec = np.repeat(vec, 2, 0)  # 320 PPG -> 160 * 2
        ppg = np.load(ppg)
        ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
        spk = np.load(spk)

        pit = torch.FloatTensor(pit)
        vec = torch.FloatTensor(vec)
        ppg = torch.FloatTensor(ppg)
        spk = torch.FloatTensor(spk)

        len_pit = pit.size()[0]
        len_vec = vec.size()[0] - 2 # for safe
        len_ppg = ppg.size()[0] - 2 # for safe
        len_min = min(len_pit, len_vec)
        len_min = min(len_min, len_ppg)
        len_wav = len_min * self.hop_length

        pit = pit[:len_min]
        vec = vec[:len_min, :]
        ppg = ppg[:len_min, :]
        spe = spe[:, :len_min]
        wav = wav[:, :len_wav]
        if len_min > use:
            max_frame_start = ppg.size(0) - use - 1
            frame_start = random.randint(0, max_frame_start)
            frame_end = frame_start + use

            pit = pit[frame_start:frame_end]
            vec = vec[frame_start:frame_end, :]
            ppg = ppg[frame_start:frame_end, :]
            spe = spe[:, frame_start:frame_end]

            wav_start = frame_start * self.hop_length
            wav_end = frame_end * self.hop_length
            wav = wav[:, wav_start:wav_end]
        # print(spe.shape)
        # print(wav.shape)
        # print(ppg.shape)
        # print(pit.shape)
        # print(spk.shape)
        return spe, wav, ppg, vec, pit, spk


class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        # mel: [freq, length]
        # wav: [1, length]
        # ppg: [len, 1024]
        # pit: [len]
        # spk: [256]
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
        )

        max_spe_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[1].size(1) for x in batch])
        spe_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        spe_padded = torch.FloatTensor(
            len(batch), batch[0][0].size(0), max_spe_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        spe_padded.zero_()
        wav_padded.zero_()

        max_ppg_len = max([x[2].size(0) for x in batch])
        ppg_lengths = torch.FloatTensor(len(batch))
        ppg_padded = torch.FloatTensor(
            len(batch), max_ppg_len, batch[0][2].size(1))
        vec_padded = torch.FloatTensor(
            len(batch), max_ppg_len, batch[0][3].size(1))
        pit_padded = torch.FloatTensor(len(batch), max_ppg_len)
        ppg_padded.zero_()
        vec_padded.zero_()
        pit_padded.zero_()
        spk = torch.FloatTensor(len(batch), batch[0][5].size(0))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            spe = row[0]
            spe_padded[i, :, : spe.size(1)] = spe
            spe_lengths[i] = spe.size(1)

            wav = row[1]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            ppg = row[2]
            ppg_padded[i, : ppg.size(0), :] = ppg
            ppg_lengths[i] = ppg.size(0)

            vec = row[3]
            vec_padded[i, : vec.size(0), :] = vec

            pit = row[4]
            pit_padded[i, : pit.size(0)] = pit

            spk[i] = row[5]
        # print(ppg_padded.shape)
        # print(ppg_lengths.shape)
        # print(pit_padded.shape)
        # print(spk.shape)
        # print(spe_padded.shape)
        # print(spe_lengths.shape)
        # print(wav_padded.shape)
        # print(wav_lengths.shape)
        return (
            ppg_padded,
            ppg_lengths,
            vec_padded,
            pit_padded,
            spk,
            spe_padded,
            spe_lengths,
            wav_padded,
            wav_lengths,
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(
                    len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            if (len_bucket == 0):
                continue
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank:: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size: (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
