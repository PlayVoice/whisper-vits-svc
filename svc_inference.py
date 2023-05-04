import os
import torch
import librosa
import argparse
import numpy as np
import torchcrepe

from omegaconf import OmegaConf
from scipy.io.wavfile import write
from vits.models import SynthesizerInfer


def load_svc_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    saved_state_dict = checkpoint_dict["model_g"]
    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = saved_state_dict[k]
    model.load_state_dict(new_state_dict)
    return model


def compute_f0_nn(filename, device):
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    # Load audio
    audio = torch.tensor(np.copy(audio))[None]
    # Here we'll use a 20 millisecond hop length
    hop_length = 320
    # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
    # This would be a reasonable range for speech
    fmin = 50
    fmax = 1000
    # Select a model capacity--one of "tiny" or "full"
    model = "full"
    # Pick a batch size that doesn't cause memory errors on your gpu
    batch_size = 512
    # Compute pitch using first gpu
    pitch, periodicity = torchcrepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=True,
    )
    pitch = np.repeat(pitch, 2, -1)  # 320 -> 160 * 2
    periodicity = np.repeat(periodicity, 2, -1)  # 320 -> 160 * 2
    # CREPE was not trained on silent audio. some error on silent need filter.
    periodicity = torchcrepe.filter.median(periodicity, 9)
    pitch = torchcrepe.filter.mean(pitch, 9)
    pitch[periodicity < 0.1] = 0
    pitch = pitch.squeeze(0)
    return pitch


def main(args):
    if (args.ppg == None):
        args.ppg = "svc_tmp.ppg.npy"
        print(
            f"Auto run : python whisper/inference.py -w {args.wave} -p {args.ppg}")
        os.system(f"python whisper/inference.py -w {args.wave} -p {args.ppg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = OmegaConf.load(args.config)
    model = SynthesizerInfer(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp)
    load_svc_model(args.model, model)
    model.eval()
    model.to(device)

    spk = np.load(args.spk)
    spk = torch.FloatTensor(spk)

    ppg = np.load(args.ppg)
    ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
    ppg = torch.FloatTensor(ppg)

    pit = compute_f0_nn(args.wave, device)
    if (args.statics == None):
        print("don't use pitch shift")
    else:
        source = pit[pit > 0]
        source_ave = source.mean()
        source_min = source.min()
        source_max = source.max()
        print(f"source pitch statics: mean={source_ave:0.1f}, \
                min={source_min:0.1f}, max={source_max:0.1f}")
        singer_ave, singer_min, singer_max = np.load(args.statics)
        print(f"singer pitch statics: mean={singer_ave:0.1f}, \
                min={singer_min:0.1f}, max={singer_max:0.1f}")

        shift = np.log2(singer_ave/source_ave) * 12
        if (singer_ave >= source_ave):
            shift = np.floor(shift)
        else:
            shift = np.ceil(shift)
        shift = 2 ** (shift / 12)
        pit = pit * shift

    pit = torch.FloatTensor(pit)

    len_pit = pit.size()[0]
    len_ppg = ppg.size()[0]
    len_min = min(len_pit, len_ppg)
    pit = pit[:len_min]
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
        has_audio = False

        while (out_index + out_chunk < all_frame):
            has_audio = True
            if (out_index == 0):  # start frame
                cut_s = 0
                cut_s_48k = 0
            else:
                cut_s = out_index - hop_frame
                cut_s_48k = hop_frame * hop_size

            if (out_index + out_chunk + hop_frame > all_frame):  # end frame
                cut_e = out_index + out_chunk
                cut_e_48k = 0
            else:
                cut_e = out_index + out_chunk + hop_frame
                cut_e_48k = -1 * hop_frame * hop_size

            sub_ppg = ppg[cut_s:cut_e, :].unsqueeze(0).to(device)
            sub_pit = pit[cut_s:cut_e].unsqueeze(0).to(device)
            sub_len = torch.LongTensor([cut_e - cut_s]).to(device)
            sub_har = source[:, :, cut_s *
                             hop_size:cut_e * hop_size].to(device)
            sub_out = model.inference(sub_ppg, sub_pit, spk, sub_len, sub_har)
            sub_out = sub_out[0, 0].data.cpu().detach().numpy()

            sub_out = sub_out[cut_s_48k:cut_e_48k]
            out_audio.extend(sub_out)
            out_index = out_index + out_chunk

        if (out_index < all_frame):
            if (has_audio):
                cut_s = out_index - hop_frame
                cut_s_48k = hop_frame * hop_size
            else:
                cut_s = 0
                cut_s_48k = 0
            sub_ppg = ppg[cut_s:, :].unsqueeze(0).to(device)
            sub_pit = pit[cut_s:].unsqueeze(0).to(device)
            sub_len = torch.LongTensor([all_frame - cut_s]).to(device)
            sub_har = source[:, :, cut_s * hop_size:].to(device)
            sub_out = model.inference(sub_ppg, sub_pit, spk, sub_len, sub_har)
            sub_out = sub_out[0, 0].data.cpu().detach().numpy()

            sub_out = sub_out[cut_s_48k:]
            out_audio.extend(sub_out)
        out_audio = np.asarray(out_audio)

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
    parser.add_argument('--statics', type=str,
                        help="Path of pitch statics.")
    args = parser.parse_args()

    main(args)
