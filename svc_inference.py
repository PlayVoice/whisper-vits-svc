import logging
import sys,os
from pathlib import Path

import faiss

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import argparse
import numpy as np

from omegaconf import OmegaConf
from scipy.io.wavfile import write
from vits.models import SynthesizerInfer
from pitch import load_csv_pitch


logger = logging.getLogger(__name__)


class IndexRetrieval:
    def __init__(self, ratio: float, n_nearest: int, hubert_index, whisper_index) -> None:
        logger.debug("init faiss retrival index with params: ratio=%s n_nearest=%s", ratio, n_nearest)
        self.ratio = ratio
        self.n_nearest = n_nearest
        self.hubert_index = hubert_index
        self.whisper_index = whisper_index

    def _create_retriv_vector(self, source_vectors, nearest_vectors, scores):
        """
        source_vectors dim (num_vectors, vector_dim)
        nearest_vectors dim (n_nearest, vector_dim)
        scores dim (num_vectors, n_nearest)
        """
        # use magic code from original RVC
        # https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/86ed98aacaa8b2037aad795abd11cdca122cf39f/vc_infer_pipeline.py#L213C18-L213C19
        logger.debug("shape: sv=%s nv=%s sc=%s", source_vectors.shape, nearest_vectors.shape, scores.shape)
        weight = np.square(1 / scores)
        weight /= weight.sum(axis=1, keepdims=True)
        weight = np.expand_dims(weight, axis=2)
        logger.debug("shape: nv=%s weight=%s", nearest_vectors.shape, weight.shape)
        weighted_nearest_vectors = np.sum(nearest_vectors * weight, axis=1)
        retriv_vector = (1 - self.ratio) * source_vectors + self.ratio * weighted_nearest_vectors
        return retriv_vector

    def _retriv_by_index(self, index, vec: torch.Tensor) -> torch.Tensor:
        np_vec = vec.numpy()
        # use method search_and_reconstruct instead of recreating the whole matrix
        scores, _, nearest_vectors = index.search_and_reconstruct(np_vec, k=self.n_nearest)
        retriv = self._create_retriv_vector(np_vec, nearest_vectors, scores)
        return torch.from_numpy(retriv)

    def _retriv_whisper(self, vec):
        logger.debug("start retriv whisper")
        return self._retriv_by_index(self.whisper_index, vec)

    def _retriv_hubert(self, vec):
        logger.debug("start retriv hubert")
        return self._retriv_by_index(self.hubert_index, vec)

    def retriv(self, hubert_vec, whisper_vec):
        logger.debug("start retriv")
        return self._retriv_hubert(hubert_vec), self._retriv_whisper(whisper_vec)


class DummyRetrieval:
    def retriv(self, hubert_vec, whisper_vec):
        return hubert_vec, whisper_vec


def load_hubert_index(base_path: Path, speaker: str):
    index_filepath = base_path / "data_svc" / "indexes" / speaker / "hubert.index"
    return faiss.read_index(str(index_filepath))


def load_whisper_index(base_path: Path, speaker: str):
    index_filepath = base_path / "data_svc" / "indexes" / speaker / "whisper.index"
    return faiss.read_index(str(index_filepath))


def get_speaker_name_from_path(speaker_path: Path) -> str:
    suffixes = "".join(speaker_path.suffixes)
    filename = speaker_path.name
    return filename.rstrip(suffixes)


def create_retrival(cli_args):
    if not cli_args.enable_retrieval:
        logger.info("infer without retrival")
        return DummyRetrieval()
    else:
        logger.info("load index retrival model")

    if 0 > cli_args.retrieval_ratio > 1:
        raise ValueError("retrieval-ratio must be in range 0..1")

    if 1 > cli_args.n_retrieval_vectors:
        raise ValueError("n-retrieval-vectors must be gte 1")

    base_path = Path(".").absolute()
    speaker_name = get_speaker_name_from_path(Path(args.spk))
    return IndexRetrieval(
        ratio=cli_args.retrieval_ratio,
        n_nearest=cli_args.n_retrieval_vectors,
        hubert_index=load_hubert_index(base_path, speaker_name),
        whisper_index=load_whisper_index(base_path, speaker_name),
    )


def load_svc_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    saved_state_dict = checkpoint_dict["model_g"]
    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            print("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model


def svc_infer(model, retrieval: IndexRetrieval, spk, pit, ppg, vec, hp, device):
    len_pit = pit.size()[0]
    len_vec = vec.size()[0]
    len_ppg = ppg.size()[0]
    len_min = min(len_pit, len_vec)
    len_min = min(len_min, len_ppg)
    pit = pit[:len_min]
    vec = vec[:len_min, :]
    ppg = ppg[:len_min, :]

    with torch.no_grad():
        spk = spk.unsqueeze(0).to(device)
        source = pit.unsqueeze(0).to(device)
        source = model.pitch2source(source)
        pitwav = model.source2wav(source)
        write("svc_out_pit.wav", hp.data.sampling_rate, pitwav)

        hop_size = hp.data.hop_length
        all_frame = len_min
        hop_frame = 10
        out_chunk = 2500  # 25 S
        out_index = 0
        out_audio = []

        while (out_index < all_frame):

            if (out_index == 0):  # start frame
                cut_s = 0
                cut_s_out = 0
            else:
                cut_s = out_index - hop_frame
                cut_s_out = hop_frame * hop_size

            if (out_index + out_chunk + hop_frame > all_frame):  # end frame
                cut_e = all_frame
                cut_e_out = -1
            else:
                cut_e = out_index + out_chunk + hop_frame
                cut_e_out = -1 * hop_frame * hop_size

            sub_ppg = ppg[cut_s:cut_e, :].unsqueeze(0).to(device)
            sub_vec = vec[cut_s:cut_e, :].unsqueeze(0).to(device)
            sub_vec, sub_ppg = retrieval.retriv(sub_vec, sub_ppg)
            sub_pit = pit[cut_s:cut_e].unsqueeze(0).to(device)
            sub_len = torch.LongTensor([cut_e - cut_s]).to(device)
            sub_har = source[:, :, cut_s *
                             hop_size:cut_e * hop_size].to(device)
            sub_out = model.inference(
                sub_ppg, sub_vec, sub_pit, spk, sub_len, sub_har)
            sub_out = sub_out[0, 0].data.cpu().detach().numpy()

            sub_out = sub_out[cut_s_out:cut_e_out]
            out_audio.extend(sub_out)
            out_index = out_index + out_chunk

        out_audio = np.asarray(out_audio)
    return out_audio


def main(args):
    if (args.ppg == None):
        args.ppg = "svc_tmp.ppg.npy"
        print(
            f"Auto run : python whisper/inference.py -w {args.wave} -p {args.ppg}")
        os.system(f"python whisper/inference.py -w {args.wave} -p {args.ppg}")

    if (args.vec == None):
        args.vec = "svc_tmp.vec.npy"
        print(
            f"Auto run : python hubert/inference.py -w {args.wave} -v {args.vec}")
        os.system(f"python hubert/inference.py -w {args.wave} -v {args.vec}")

    if (args.pit == None):
        args.pit = "svc_tmp.pit.csv"
        print(
            f"Auto run : python pitch/inference.py -w {args.wave} -p {args.pit}")
        os.system(f"python pitch/inference.py -w {args.wave} -p {args.pit}")

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = OmegaConf.load(args.config)
    model = SynthesizerInfer(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp)
    load_svc_model(args.model, model)
    retrieval = create_retrival(args)
    model.eval()
    model.to(device)

    spk = np.load(args.spk)
    spk = torch.FloatTensor(spk)

    ppg = np.load(args.ppg)
    ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
    ppg = torch.FloatTensor(ppg)
    # ppg = torch.zeros_like(ppg)

    vec = np.load(args.vec)
    vec = np.repeat(vec, 2, 0)  # 320 PPG -> 160 * 2
    vec = torch.FloatTensor(vec)
    # vec = torch.zeros_like(vec)

    pit = load_csv_pitch(args.pit)
    print("pitch shift: ", args.shift)
    if (args.shift == 0):
        pass
    else:
        pit = np.array(pit)
        source = pit[pit > 0]
        source_ave = source.mean()
        source_min = source.min()
        source_max = source.max()
        print(f"source pitch statics: mean={source_ave:0.1f}, \
                min={source_min:0.1f}, max={source_max:0.1f}")
        shift = args.shift
        shift = 2 ** (shift / 12)
        pit = pit * shift
    pit = torch.FloatTensor(pit)

    out_audio = svc_infer(model, retrieval, spk, pit, ppg, vec, hp, device)
    write("svc_out.wav", hp.data.sampling_rate, out_audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('--model', type=str, required=True,
                        help="path of model for evaluation")
    parser.add_argument('--wave', type=str, required=True,
                        help="Path of raw audio.")
    parser.add_argument('--spk', type=str, required=True,
                        help="Path of speaker.")
    parser.add_argument('--ppg', type=str,
                        help="Path of content vector.")
    parser.add_argument('--vec', type=str,
                        help="Path of hubert vector.")
    parser.add_argument('--pit', type=str,
                        help="Path of pitch csv file.")
    parser.add_argument('--shift', type=int, default=0,
                        help="Pitch shift key.")
    parser.add_argument('--enable-retrieval', action="store_true",
                        help="Enable index feature retrieval")
    parser.add_argument('--retrieval-ratio', type=float, default=.5,
                        help="ratio of feature retrieval effect. Must be in range 0..1")
    parser.add_argument('--n-retrieval-vectors', type=int, default=3, choices=[1, 2, 3],
                        help="get n nearest vectors from retrieval index. Must be in range 1..3")
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()

    main(args)
