import argparse
import numpy as np


def save_csv_pitch(pitch, path):
    with open(path, "w", encoding='utf-8') as pitch_file:
        for i in range(len(pitch)):
            t = i * 10
            minute = t // 60000
            seconds = (t - minute * 60000) // 1000
            millisecond = t % 1000
            print(
                f"{minute}m {seconds}s {millisecond:3d},{int(pitch[i])}", file=pitch_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-p", "--pit", help="pit", dest="pit")  # pit for train
    args = parser.parse_args()
    print(args.pit)

    pitch = np.load(args.pit)
    save_csv_pitch(pitch, 'pitch_debug.csv')
