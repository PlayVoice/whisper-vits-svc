# -*- coding: utf-8 -*-
# %%
import argparse
import numpy as np
from pathlib import Path
from .model import *
from .featureExtraction import *
from .quantization import *
from .utils import *
from .MIDI import *

# %%
class SingingTranscription:
    def __init__(self):

        self.PATH_PROJECT = pathlib.Path(__file__).absolute().parent.parent
        self.num_spec = 513
        self.window_size = 31
        self.note_res = 1
        self.batch_size = 64

    def load_model(self, path_weight, TF_summary=False):

        model = melody_ResNet_JDC(self.num_spec, self.window_size, self.note_res)
        model.load_weights(path_weight)
        if TF_summary == True:
            print(model.summary())
        return model

    def predict_melody(self, model_ST, filepath):
        pitch_range = np.arange(40, 95 + 1.0 / self.note_res, 1.0 / self.note_res)
        pitch_range = np.concatenate([np.zeros(1), pitch_range])

        """  Features extraction"""
        X_test, _ = spec_extraction(file_name=filepath, win_size=self.window_size)

        """  melody predict"""
        y_predict = model_ST.predict(X_test, batch_size=self.batch_size, verbose=1)
        y_predict = y_predict[0]  # [0]:note,  [1]:vocing
        y_shape = y_predict.shape
        num_total = y_shape[0] * y_shape[1]
        y_predict = np.reshape(y_predict, (num_total, y_shape[2]))

        est_MIDI = np.zeros(num_total)
        est_freq = np.zeros(num_total)
        for i in range(num_total):
            index_predict = np.argmax(y_predict[i])
            pitch_MIDI = pitch_range[np.int32(index_predict)]
            if pitch_MIDI >= 40 and pitch_MIDI <= 95:
                est_MIDI[i] = pitch_MIDI
                # est_freq[i] = 2 ** ((pitch_MIDI - 69) / 12.0) * 440
        return est_MIDI

    def save_output_frame_level(self, pitch_score, path_save, note_or_freq="note"):
        check_and_make_dir(Path(path_save))
        f = open(path_save, "w")

        assert (note_or_freq == "freq") or (note_or_freq == "note"), "please check 'note' or 'freq"
        if note_or_freq == "freq":
            for j in range(len(pitch_score)):
                if pitch_score[j] > 0:
                    pitch_score[j] = 2 ** ((pitch_score[j] - 69) / 12.0) * 440
                est = "%.2f %.4f\n" % (0.01 * j, pitch_score[j])
                f.write(est)
        elif note_or_freq == "note":
            for j in range(len(pitch_score)):
                est = "%.2f %.4f\n" % (0.01 * j, pitch_score[j])
                f.write(est)

        f.close()


def main(args):
    ST = SingingTranscription()

    """ load model """
    model_ST = ST.load_model(f"{ST.PATH_PROJECT}/data/weight_ST.hdf5", TF_summary=False)

    """ predict note (time-freq) """
    path_audio = args.path_audio
    fl_note = ST.predict_melody(model_ST, path_audio)  # frame-level pitch score

    """ post-processing """
    tempo = calc_tempo(path_audio)
    refined_fl_note = refine_note(fl_note, tempo)  # frame-level pitch score

    """ convert frame-level pitch score to note-level (time-axis) """
    segment = note_to_segment(refined_fl_note)  # note-level pitch score

    """ save ouput to .mid """
    filename = get_filename_wo_extension(path_audio)
    path_output = f"{args.path_save}/{filename}.mid"
    segment_to_midi(segment, path_output=path_output, tempo=tempo)

    if args.output_type == "fps":
        path_note = f"{args.path_save}/{filename}.txt"
        ST.save_output_frame_level(refined_fl_note, path_note, note_or_freq="freq")

    print(f"\n========= DONE =========")
    print(f"input: '{path_audio}'")
    print(f"output: '{path_output}'")

ST = SingingTranscription()

""" load model """
model_ST = ST.load_model(f"{ST.PATH_PROJECT}/data/weight_ST.hdf5", TF_summary=False)


def get_frame_level_output(wav_path):
    """ predict note (time-freq) """
    path_audio = wav_path
    fl_note = ST.predict_melody(model_ST, path_audio)  # frame-level pitch score

    """ post-processing """
    tempo = calc_tempo(path_audio)
    refined_fl_note = refine_note(fl_note, tempo)  # frame-level pitch score
    return refined_fl_note.astype(int)


# %%
if __name__ == "__main__":
    PATH_PROJECT = pathlib.Path(__file__).absolute().parent.parent
    parser = argparse.ArgumentParser(description="Predict singing transcription")
    parser.add_argument(
        "-i",
        "--path_audio",
        type=str,
        help="Path to input audio file.",
        default=f"{PATH_PROJECT}/audio/pop1.wav",
    )
    parser.add_argument(
        "-o",
        "--path_save",
        type=str,
        help="Path to folder for saving .mid file",
        default=f"{PATH_PROJECT}/output",
    )

    parser.add_argument(
        "-ot",
        "--output_type",
        type=str,
        help="(optional) Output type: midi or frame-level pitch score(fps)",
        default="midi",
    )

    main(parser.parse_args())
