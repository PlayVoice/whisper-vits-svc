import argparse
import subprocess
import signal

parser = argparse.ArgumentParser()
parser.add_argument("-t", type=int, default=0, help="thread count")
parser.add_argument("--crepe", action="store_true", help="Use crepe to extract f0")
args = parser.parse_args()

if args.crepe:
   commands = [
      "python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-16k -s 16000 -t 0",
      "python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-32k -s 32000 -t 0",
      "python prepare/preprocess_f0_crepe.py -w data_svc/waves-16k/ -p data_svc/pitch -t "+str(args.t),
      "python prepare/preprocess_ppg.py -w data_svc/waves-16k/ -p data_svc/whisper -t 1",
      "python prepare/preprocess_speaker.py data_svc/waves-16k/ data_svc/speaker -t 0",
      "python prepare/preprocess_speaker_ave.py data_svc/speaker/ data_svc/singer -t 0",
      "python prepare/preprocess_spec.py -w data_svc/waves-32k/ -s data_svc/specs -t 0",
      "python prepare/preprocess_train.py -t 0",
      "python prepare/preprocess_zzz.py",
   ]

else:
   commands = [
      "python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-16k -s 16000 -t 0",
      "python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-32k -s 32000 -t 0",
      "python prepare/preprocess_f0.py -w data_svc/waves-16k/ -p data_svc/pitch -t "+str(args.t),
      "python prepare/preprocess_ppg.py -w data_svc/waves-16k/ -p data_svc/whisper -t 1",
      "python prepare/preprocess_speaker.py data_svc/waves-16k/ data_svc/speaker -t 0",
      "python prepare/preprocess_speaker_ave.py data_svc/speaker/ data_svc/singer -t 0",
      "python prepare/preprocess_spec.py -w data_svc/waves-32k/ -s data_svc/specs -t 0",
      "python prepare/preprocess_train.py -t 0",
      "python prepare/preprocess_zzz.py",
   ]

for command in commands:
   print(f"Command: {command}")
   process = subprocess.Popen(command, shell=True)
   process.wait()
