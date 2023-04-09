import re
import json
import fsspec
import torch
import numpy as np
import argparse

from argparse import RawTextHelpFormatter
from .models.lstm import LSTMSpeakerEncoder
from .config import SpeakerEncoderConfig
from .utils.audio import AudioProcessor


def read_json(json_path):
    config_dict = {}
    try:
        with fsspec.open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.decoder.JSONDecodeError:
        # backwards compat.
        data = read_json_with_comments(json_path)
    config_dict.update(data)
    return config_dict


def read_json_with_comments(json_path):
    """for backward compat."""
    # fallback to json
    with fsspec.open(json_path, "r", encoding="utf-8") as f:
        input_str = f.read()
    # handle comments
    input_str = re.sub(r"\\\n", "", input_str)
    input_str = re.sub(r"//.*\n", "\n", input_str)
    data = json.loads(input_str)
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""Compute embedding vectors for each wav file in a dataset.""",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("model_path", type=str, help="Path to model checkpoint file.")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to model config file.",
    )

    parser.add_argument("-s", "--source", help="input wave", dest="source")
    parser.add_argument(
        "-t", "--target", help="output 256d speaker embeddimg", dest="target"
    )

    parser.add_argument("--use_cuda", type=bool, help="flag to set cuda.", default=True)
    parser.add_argument("--eval", type=bool, help="compute eval.", default=True)

    args = parser.parse_args()
    source_file = args.source
    target_file = args.target

    # config
    config_dict = read_json(args.config_path)
    # print(config_dict)

    # model
    config = SpeakerEncoderConfig(config_dict)
    config.from_dict(config_dict)

    speaker_encoder = LSTMSpeakerEncoder(
        config.model_params["input_dim"],
        config.model_params["proj_dim"],
        config.model_params["lstm_dim"],
        config.model_params["num_lstm_layers"],
    )

    speaker_encoder.load_checkpoint(args.model_path, eval=True, use_cuda=args.use_cuda)

    # preprocess
    speaker_encoder_ap = AudioProcessor(**config.audio)
    # normalize the input audio level and trim silences
    speaker_encoder_ap.do_sound_norm = True
    speaker_encoder_ap.do_trim_silence = True

    # compute speaker embeddings

    # extract the embedding
    waveform = speaker_encoder_ap.load_wav(
        source_file, sr=speaker_encoder_ap.sample_rate
    )
    spec = speaker_encoder_ap.melspectrogram(waveform)
    spec = torch.from_numpy(spec.T)
    if args.use_cuda:
        spec = spec.cuda()
    spec = spec.unsqueeze(0)
    embed = speaker_encoder.compute_embedding(spec).detach().cpu().numpy()
    embed = embed.squeeze()
    # print(embed)
    # print(embed.size)
    np.save(target_file, embed, allow_pickle=False)


    if hasattr(speaker_encoder, 'module'):
        state_dict = speaker_encoder.module.state_dict()
    else:
        state_dict = speaker_encoder.state_dict()
        torch.save({'model': state_dict}, "model_small.pth")
