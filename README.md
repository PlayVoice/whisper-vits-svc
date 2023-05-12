# Singing Voice Conversion
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/maxmax20160403/sovits5.0)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PY1E4bDAeHbAD4r99D_oYXB46fG8nIA5?usp=sharing)
<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/PlayVoice/so-vits-svc-5.0">
<img alt="GitHub forks" src="https://img.shields.io/github/forks/PlayVoice/so-vits-svc-5.0">
<img alt="GitHub issues" src="https://img.shields.io/github/issues/PlayVoice/so-vits-svc-5.0">
<img alt="GitHub" src="https://img.shields.io/github/license/PlayVoice/so-vits-svc-5.0">

![sovits_framework](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/402cf58d-6d03-4d0b-9d6a-94f079898672)

【无 泄漏】支持多发音人的SVC库

【带 伴奏】也能进行歌声转换的SVC库（轻度伴奏）

【用 Excel】进行原始的SVC手工调教

![sonic visualiser](https://user-images.githubusercontent.com/16432329/237011482-51f3a45e-72c6-4d4a-b1df-f561d1df7132.png)

## 本项目与svc-develop-team/so-vits-svc仓库的关系

svc-develop-team/so-vits-svc基于PlayVoice/VI-SVC演变而来，见https://github.com/svc-develop-team/so-vits-svc/tree/2.0

本项目是PlayVoice/VI-SVC的继续完善，而非基于svc-develop-team/so-vits-svc

## 本项目预览模型已发布，还需要更多的时间训练到最佳状态

- 预览模型包括：生成器+判别器=194M，设置batch_size为8时，训练占用7.5G显存，学习门槛大大降低
- 预览模型包含56个发音人，发音人文件在configs/singers目录中，可进行推理测试，尤其测试音色泄露
- 发音人22，30，47，51辨识度较高，发音人样本在configs/singers_sample目录中

| Feature | From | Status | Function | Remarks |
| --- | --- | --- | --- | --- |
| whisper | OpenAI | ✅ | 强大的抗噪能力 | 必须 |
| bigvgan  | NVIDA | ✅ | 抗锯齿与蛇形激活 | 删除，GPU占用过多 |
| natural speech | Microsoft | ✅ | 减少发音错误 | 二阶段训练 |
| neural source-filter | NII | ✅ | 解决断音问题 | 必须 |
| speaker encoder | Google | ✅ | 音色编码与聚类 | 必须 |
| GRL for speaker | Skoltech |✅ |防止编码器泄露音色 | 二阶段训练 |
| one shot vits |  Samsung | ✅ | VITS 一句话克隆 | 必须 |
| SCLN |  Microsoft | ✅ | 改善克隆 | 必须 |
| band extention | Adobe | ✅ | 16K升48K采样 | 数据处理 |

## 模型简介
歌声音色转换模型，通过SoftVC内容编码器提取源音频语音特征，与F0同时输入VITS替换原本的文本输入达到歌声转换的效果。同时，更换声码器为 [NSF HiFiGAN](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan) 解决断音问题

> 据不完全统计，多说话人似乎会导致**音色泄漏加重**，不建议训练超过10人的模型，目前的建议是如果想炼出来更像目标音色，**尽可能炼单说话人的**\
> 针对sovits3.0 48khz模型推理显存占用大的问题，可以切换到[32khz的分支](https://github.com/innnky/so-vits-svc/tree/32k) 版本训练32khz的模型\
> 目前发现一个较大问题，3.0推理时显存占用巨大，6G显存基本只能推理30s左右长度音频\
> 断音问题已解决，音质提升了不少\
> 2.0版本已经移至 sovits_2.0分支\
> 3.0版本使用FreeVC的代码结构，与旧版本不通用\
> 与[DiffSVC](https://github.com/prophesier/diff-svc) 相比，在训练数据质量非常高时diffsvc有着更好的表现，对于质量差一些的数据集，本仓库可能会有更好的表现，此外，本仓库推理速度上比diffsvc快很多


## 数据集准备

💗必要的前处理：
- 1 降噪&去伴奏
- 2 频率提升
- 3 音质提升，基于https://github.com/openvpi/vocoders ，待整合

然后以下面文件结构将数据集放入dataset_raw目录
```shell
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```

## 安装依赖

- 1 软件依赖

  > pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

- 2 下载音色编码器: [Speaker-Encoder by @mueller91](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3), 解压文件，把 `best_model.pth.tar`  放到目录 `speaker_pretrain/`

- 3 下载whisper模型 [multiple language medium model](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), 确定下载的是`medium.pt`，把它放到文件夹 `whisper_pretrain/`


## 数据预处理
- 1， 设置工作目录:heartpulse::heartpulse::heartpulse:不设置后面会报错

    > export PYTHONPATH=$PWD

- 2， 重采样

    将音频剪裁为小于30秒的音频段，whisper的要求

    生成采样率16000Hz音频, 存储路径为：./data_svc/waves-16k

    > python prepare/preprocess_a.py -w ./data_raw -o ./data_svc/waves-16k -s 16000

    生成采样率32000Hz音频, 存储路径为：./data_svc/waves-32k

    > python prepare/preprocess_a.py -w ./data_raw -o ./data_svc/waves-32k -s 32000

    可选的16000Hz提升到32000Hz，待完善~批处理

    > python bandex/inference.py -w svc_out.wav

- 3， 使用16K音频，提取音高
    > python prepare/preprocess_f0.py -w data_svc/waves-16k/ -p data_svc/pitch

- 4， 使用16k音频，提取内容编码
    > python prepare/preprocess_ppg.py -w data_svc/waves-16k/ -p data_svc/whisper

- 5， 使用16k音频，提取音色编码
    > python prepare/preprocess_speaker.py data_svc/waves-16k/ data_svc/speaker

- 6， 使用32k音频，提取线性谱
    > python prepare/preprocess_spec.py -w data_svc/waves-32k/ -s data_svc/specs

- 7， 使用32k音频，生成训练索引
    > python prepare/preprocess_train.py

- 8， 训练文件调试
    > python prepare/preprocess_zzz.py


## 训练
- 1， 设置工作目录:heartpulse::heartpulse::heartpulse:不设置后面会报错

    > export PYTHONPATH=$PWD

- 2， 启动训练，一阶段训练

    > python svc_trainer.py -c configs/base.yaml -n sovits5.0

- 3， 恢复训练

    > python svc_trainer.py -c configs/base.yaml -n sovits5.0 -p chkpt/sovits5.0/***.pth

- 4， 查看日志，release页面有完整的训练日志

    > tensorboard --logdir logs/

- 5， 启动训练，二阶段训练:heartpulse:

    二阶段训练内容：PPG扰动，GRL去音色，natural speech推理loss;验证中~~~

    > python svc_trainer.py -c configs/more.yaml -n more -e 1

20K一阶段训练日志如下，可以看到还未收敛完成

![sovits5 0 preview](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/339c11d5-67dd-426a-ba19-077d66efc953)

![sovits_spec](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/c4223cf3-b4a0-4325-bec0-6d46d195a1fc)

## 推理

- 1， 设置工作目录:heartpulse::heartpulse::heartpulse:不设置后面会报错

    > export PYTHONPATH=$PWD

- 2， 导出推理模型：文本编码器，Flow网络，Decoder网络；判别器和后验编码器只在训练中使用

    > python svc_export.py --config configs/base.yaml --checkpoint_path chkpt/sovits5.0/***.pt

- 3， 使用whisper提取内容编码，没有采用一键推理，为了降低显存占用

    > python whisper/inference.py -w test.wav -p test.ppg.npy

    生成test.ppg.npy；如果下一步没有指定ppg文件，则调用程序自动生成

- 4， 提取csv文本格式F0参数，Excel打开csv文件，对照Audition或者SonicVisualiser手动修改错误的F0

    > python pitch/inference.py -w test.wav -p test.csv

![Audition ](https://user-images.githubusercontent.com/16432329/237006512-9ef97936-df00-4b2d-ab76-921c383eb616.png)

- 5，指定参数，推理

    > python svc_inference.py --config configs/base.yaml --model sovits5.0.pth --spk ./configs/singers/singer0001.npy --wave test.wav --ppg test.ppg.npy --pit test.csv

    当指定--ppg后，多次推理同一个音频时，可以避免重复提取音频内容编码；没有指定，也会自动提取；

    当指定--pit后，可以加载手工调教的F0参数；没有指定，也会自动提取；

    生成文件在当前目录svc_out.wav；

    | args |--config | --model | --spk | --wave | --ppg | --pit |          
    | ---  | --- | --- | --- | --- | --- | --- |
    | name | 配置文件 | 模型文件 | 音色文件 | 音频文件 | 音频内容 | 音高内容 |

## 数据集

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

## 代码来源和参考文献

https://github.com/facebookresearch/speech-resynthesis [paper](https://arxiv.org/abs/2104.00355)

https://github.com/jaywalnut310/vits [paper](https://arxiv.org/abs/2106.06103)

https://github.com/openai/whisper/ [paper](https://arxiv.org/abs/2212.04356)

https://github.com/NVIDIA/BigVGAN [paper](https://arxiv.org/abs/2206.04658)

https://github.com/mindslab-ai/univnet [[paper]](https://arxiv.org/abs/2106.07889)

https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

https://github.com/brentspell/hifi-gan-bwe

https://github.com/mozilla/TTS

[SNAC : Speaker-normalized Affine Coupling Layer in Flow-based Architecture for Zero-Shot Multi-Speaker Text-to-Speech](https://github.com/hcy71o/SNAC)

[Adapter-Based Extension of Multi-Speaker Text-to-Speech Model for New Speakers](https://arxiv.org/abs/2211.00585)

[AdaSpeech: Adaptive Text to Speech for Custom Voice](https://arxiv.org/pdf/2103.00993.pdf)

[Cross-Speaker Prosody Transfer on Any Text for Expressive Speech Synthesis](https://github.com/ubisoft/ubisoft-laforge-daft-exprt)

[Learn to Sing by Listening: Building Controllable Virtual Singer by Unsupervised Learning from Voice Recordings](https://arxiv.org/abs/2305.05401)

## 贡献者

<a href="https://github.com/PlayVoice/so-vits-svc/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PlayVoice/so-vits-svc" />
</a>
