<div align="center">
<h1> Variational Inference with adversarial learning for end-to-end Singing Voice Conversion based on VITS </h1>
    
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/maxmax20160403/sovits5.0)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PY1E4bDAeHbAD4r99D_oYXB46fG8nIA5?usp=sharing)
<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/PlayVoice/so-vits-svc-5.0">
<img alt="GitHub forks" src="https://img.shields.io/github/forks/PlayVoice/so-vits-svc-5.0">
<img alt="GitHub issues" src="https://img.shields.io/github/issues/PlayVoice/so-vits-svc-5.0">
<img alt="GitHub" src="https://img.shields.io/github/license/PlayVoice/so-vits-svc-5.0">
 
</div>

- 💗This project is target for: beginners in deep learning, the basic operation of Python and PyTorch is the prerequisite for using this project;
- 💗This project aims to help deep learning beginners get rid of boring pure theoretical learning, and master the basic knowledge of deep learning by combining it with practice;
- 💗This project does not support real-time voice change; (support needs to replace whisper)
- 💗This project will not develop one-click packages for other purposes；

![sovits_framework](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/402cf58d-6d03-4d0b-9d6a-94f079898672)

- 6G memory GPU can be used to trained

- support for multiple speakers

- create unique speakers through speaker mixing

- even with light accompaniment can also be converted

- F0 can be edited using Excel

this project is not based on svc-develop-team/so-vits-svc，on the contrary，as:https://github.com/svc-develop-team/so-vits-svc/tree/2.0

## Model properties

https://github.com/PlayVoice/so-vits-svc-5.0/releases/tag/hifigan_release

- [sovits5.0_main_1500.pth](https://github.com/PlayVoice/so-vits-svc-5.0/releases/download/hifigan_release/sovits5.0_main_1500.pth) The model includes: generator + discriminator = 176M, which can be used as a pre-training model
- speakers files are in the configs/singers directory, which can be used for reasoning tests, especially for timbre leakage
- speakers 22, 30, 47, and 51 are highly recognizable, and the training audio samples are in the configs/singers_sample directory

| Feature | From | Status | Function | Remarks |
| --- | --- | --- | --- | --- |
| whisper | OpenAI | ✅ | strong noise immunity | - |
| bigvgan  | NVIDA | ✅ | alias and snake | The GPU takes up a little more, and the main branch is deleted; You need to switch to the branch [bigvgan](https://github.com/PlayVoice/so-vits-svc-5.0/tree/bigvgan)，the formant is clearer and the sound quality is obviously improved |
| natural speech | Microsoft | ✅ | reduce mispronunciation | - |
| neural source-filter | NII | ✅ | solve the problem of audio F0 discontinuity | - |
| speaker encoder | Google | ✅ | Timbre Encoding and Clustering | - |
| GRL for speaker | Ubisoft |✅ | Preventing Encoder Leakage Timbre | - |
| one shot vits |  Samsung | ✅ | Voice Clone | - |
| SCLN |  Microsoft | ✅ | Improve Clone | - |
| PPG perturbation | this project | ✅ | Improved noise immunity and de-timbre | - |
| VAE perturbation | this project | ✅ | Improve sound quality | - |

💗due to the use of data perturbation, it takes longer to train than other projects.

## Dataset preparation

Necessary pre-processing:
- 1 accompaniment separation
- 2 band extension
- 3 sound quality improvement
- 4 cut audio, less than 30 seconds for whisper💗

then put the dataset into the dataset_raw directory according to the following file structure
```shell
dataset_raw
├───speaker0
│   ├───000001.wav
│   ├───...
│   └───000xxx.wav
└───speaker1
    ├───000001.wav
    ├───...
    └───000xxx.wav
```

## Install dependencies

- 1 software dependency
  
  > apt update && sudo apt install ffmpeg
  
  > pip install -r requirements.txt

- 2 download the Timbre Encoder: [Speaker-Encoder by @mueller91](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3), put `best_model.pth.tar`  into `speaker_pretrain/`

- 3 download whisper model [multiple language medium model](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), Make sure to download `medium.pt`，put it into `whisper_pretrain/`

- 4 whisper is built-in, do not install it additionally, it will conflict and report an error

## Data preprocessing
- 1， set working directory:

    > export PYTHONPATH=$PWD

- 2， re-sampling

    generate audio with a sampling rate of 16000Hz：./data_svc/waves-16k

    > python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-16k -s 16000

    generate audio with a sampling rate of 32000Hz：./data_svc/waves-32k

    > python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-32k -s 32000

- 3， use 16K audio to extract pitch：f0_ceil=900, it needs to be modified according to the highest pitch of your data
    > python prepare/preprocess_f0.py -w data_svc/waves-16k/ -p data_svc/pitch

- 4， use 16K audio to extract ppg
    > python prepare/preprocess_ppg.py -w data_svc/waves-16k/ -p data_svc/whisper

- 5， use 16k audio to extract timbre code
    > python prepare/preprocess_speaker.py data_svc/waves-16k/ data_svc/speaker

- 6， extract the average value of the timbre code for inference; it can also replace a single audio timbre in generating the training index, and use it as the unified timbre of the speaker for training
    > python prepare/preprocess_speaker_ave.py data_svc/speaker/ data_svc/singer

- 7， use 32k audio to extract the linear spectrum
    > python prepare/preprocess_spec.py -w data_svc/waves-32k/ -s data_svc/specs

- 8， use 32k audio to generate training index
    > python prepare/preprocess_train.py

- 9， training file debugging
    > python prepare/preprocess_zzz.py

```shell
data_svc/
└── waves-16k
│    └── speaker0
│    │      ├── 000001.wav
│    │      └── 000xxx.wav
│    └── speaker1
│           ├── 000001.wav
│           └── 000xxx.wav
└── waves-32k
│    └── speaker0
│    │      ├── 000001.wav
│    │      └── 000xxx.wav
│    └── speaker1
│           ├── 000001.wav
│           └── 000xxx.wav
└── pitch
│    └── speaker0
│    │      ├── 000001.pit.npy
│    │      └── 000xxx.pit.npy
│    └── speaker1
│           ├── 000001.pit.npy
│           └── 000xxx.pit.npy
└── whisper
│    └── speaker0
│    │      ├── 000001.ppg.npy
│    │      └── 000xxx.ppg.npy
│    └── speaker1
│           ├── 000001.ppg.npy
│           └── 000xxx.ppg.npy
└── speaker
│    └── speaker0
│    │      ├── 000001.spk.npy
│    │      └── 000xxx.spk.npy
│    └── speaker1
│           ├── 000001.spk.npy
│           └── 000xxx.spk.npy
└── singer
    ├── speaker0.spk.npy
    └── speaker1.spk.npy
```

## Train
- 0， if fine-tuning based on the pre-trained model, you need to download the pre-trained model: sovits5.0_main_1500.pth

    > set pretrain: "./sovits5.0_main_1500.pth" in configs/base.yaml，and adjust the learning rate appropriately, eg 1e-5

- 1， set working directory

    > export PYTHONPATH=$PWD

- 2， start training

    > python svc_trainer.py -c configs/base.yaml -n sovits5.0

- 3， resume training

    > python svc_trainer.py -c configs/base.yaml -n sovits5.0 -p chkpt/sovits5.0/***.pth

- 4， view log

    > tensorboard --logdir logs/

![sovits5 0_base](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/1628e775-5888-4eac-b173-a28dca978faa)

## Inference

- 1， set working directory

    > export PYTHONPATH=$PWD

- 2， export inference model: text encoder, Flow network, Decoder network

    > python svc_export.py --config configs/base.yaml --checkpoint_path chkpt/sovits5.0/***.pt

- 3， use whisper to extract content encoding, without using one-click reasoning, in order to reduce GPU memory usage

    > python whisper/inference.py -w test.wav -p test.ppg.npy

    generate test.ppg.npy; if no ppg file is specified in the next step, generate it automatically

- 4， extract the F0 parameter to the csv text format, open the csv file in Excel, and manually modify the wrong F0 according to Audition or SonicVisualiser

    > python pitch/inference.py -w test.wav -p test.csv

- 5，specify parameters and infer

    > python svc_inference.py --config configs/base.yaml --model sovits5.0.pth --spk ./configs/singers/singer0001.npy --wave test.wav --ppg test.ppg.npy --pit test.csv

    when --ppg is specified, when the same audio is reasoned multiple times, it can avoid repeated extraction of audio content codes; if it is not specified, it will be automatically extracted;

    when --pit is specified, the manually tuned F0 parameter can be loaded; if not specified, it will be automatically extracted;

    generate files in the current directory:svc_out.wav

    | args |--config | --model | --spk | --wave | --ppg | --pit | --shift |
    | ---  | --- | --- | --- | --- | --- | --- | --- |
    | name | config path | model path | speaker | wave input | wave ppg | wave pitch | pitch shift |

## Creat singer
named by pure coincidence：average -> ave -> eva，eve(eva) represents conception and reproduction

> python svc_eva.py

```python
eva_conf = {
    './configs/singers/singer0022.npy': 0,
    './configs/singers/singer0030.npy': 0,
    './configs/singers/singer0047.npy': 0.5,
    './configs/singers/singer0051.npy': 0.5,
}
```

the generated singer file is：eva.spk.npy

💗both Flow and Decoder need to input timbres, and you can even input different timbre parameters to the two modules to create more unique timbres.

## Data set

| Name | URL |
| --- | --- |
|KiSing         |http://shijt.site/index.php/2021/05/16/kising-the-first-open-source-mandarin-singing-voice-synthesis-corpus/|
|PopCS          |https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/apply_form.md|
|opencpop       |https://wenet.org.cn/opencpop/download/|
|Multi-Singer   |https://github.com/Multi-Singer/Multi-Singer.github.io|
|M4Singer       |https://github.com/M4Singer/M4Singer/blob/master/apply_form.md|
|CSD            |https://zenodo.org/record/4785016#.YxqrTbaOMU4|
|KSS            |https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset|
|JVS MuSic      |https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music|
|PJS            |https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus|
|JUST Song      |https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song|
|MUSDB18        |https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems|
|DSD100         |https://sigsep.github.io/datasets/dsd100.html|
|Aishell-3      |http://www.aishelltech.com/aishell_3|
|VCTK           |https://datashare.ed.ac.uk/handle/10283/2651|

## Code sources and references

https://github.com/facebookresearch/speech-resynthesis [paper](https://arxiv.org/abs/2104.00355)

https://github.com/jaywalnut310/vits [paper](https://arxiv.org/abs/2106.06103)

https://github.com/openai/whisper/ [paper](https://arxiv.org/abs/2212.04356)

https://github.com/NVIDIA/BigVGAN [paper](https://arxiv.org/abs/2206.04658)

https://github.com/mindslab-ai/univnet [paper](https://arxiv.org/abs/2106.07889)

https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

https://github.com/brentspell/hifi-gan-bwe

https://github.com/mozilla/TTS

https://github.com/OlaWod/FreeVC [paper](https://arxiv.org/abs/2210.15418)

[SNAC : Speaker-normalized Affine Coupling Layer in Flow-based Architecture for Zero-Shot Multi-Speaker Text-to-Speech](https://github.com/hcy71o/SNAC)

[Adapter-Based Extension of Multi-Speaker Text-to-Speech Model for New Speakers](https://arxiv.org/abs/2211.00585)

[AdaSpeech: Adaptive Text to Speech for Custom Voice](https://arxiv.org/pdf/2103.00993.pdf)

[Cross-Speaker Prosody Transfer on Any Text for Expressive Speech Synthesis](https://github.com/ubisoft/ubisoft-laforge-daft-exprt)

[Learn to Sing by Listening: Building Controllable Virtual Singer by Unsupervised Learning from Voice Recordings](https://arxiv.org/abs/2305.05401)

[Adversarial Speaker Disentanglement Using Unannotated External Data for Self-supervised Representation Based Voice Conversion](https://arxiv.org/pdf/2305.09167.pdf)

[Speaker normalization (GRL) for self-supervised speech emotion recognition](https://arxiv.org/abs/2202.01252)

## Method of Preventing Timbre Leakage Based on Data Perturbation

https://github.com/auspicious3000/contentvec/blob/main/contentvec/data/audio/audio_utils_1.py

https://github.com/revsic/torch-nansy/blob/main/utils/augment/praat.py

https://github.com/revsic/torch-nansy/blob/main/utils/augment/peq.py

https://github.com/biggytruck/SpeechSplit2/blob/main/utils.py

https://github.com/OlaWod/FreeVC/blob/main/preprocess_sr.py

## Contributors

<a href="https://github.com/PlayVoice/so-vits-svc/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PlayVoice/so-vits-svc" />
</a>
