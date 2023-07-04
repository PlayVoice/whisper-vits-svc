import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-t", type=int, default=0, help="thread count")
args = parser.parse_args()


commands = [
   "python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-16k -s 16000 -t 0",
   "python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-32k -s 32000 -t 0",
   "python prepare/preprocess_crepe.py -w data_svc/waves-16k/ -p data_svc/pitch -t " + str(args.t),
   "python prepare/preprocess_ppg.py -w data_svc/waves-16k/ -p data_svc/whisper",
   "python prepare/preprocess_hubert.py -w data_svc/waves-16k/ -v data_svc/hubert -t 1",
   "python prepare/preprocess_speaker.py data_svc/waves-16k/ data_svc/speaker -t 0",
   "python prepare/preprocess_speaker_ave.py data_svc/speaker/ data_svc/singer -t 0",
   "python prepare/preprocess_spec.py -w data_svc/waves-32k/ -s data_svc/specs -t 0",
   "python prepare/preprocess_train.py",
   "python prepare/preprocess_zzz.py",
]


for command in commands:
   print(f"Command: {command}")
   process = subprocess.Popen(command, shell=True)
   process.wait()
