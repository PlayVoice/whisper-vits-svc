# Singing Voice Conversion

## 技术分支

- [PlayVoice/so-vits-svc-5.0](https://github.com/PlayVoice/so-vits-svc-5.0) 有口音，辨识度高，适合非跨语言，跨语言会口糊
  
- [PlayVoice/lora-svc](https://github.com/PlayVoice/lora-svc) 无口音，辨识度略低，适合跨语言，跨语言不口糊

- [PlayVoice/max-vc](https://github.com/PlayVoice/max-vc) 不提取F0，解决F0提取不准，丝滑的转换结果

## 本项目更新中，暂时不能使用，更新点

- 内容提取器更新为OpenAI的whisper
  
- HiFiGAN更新为NVIDA的BigVGAN
  
- VITS框架加入Micsoft的NatureSpeech优化
  
- 加入音色提取

- 加入说话人适配器

- 加入GRL阻止音色泄漏

| Feature | Status |
| --- | --- |
| whiser | ✅ |
| bigvgan  | ✅ |
| nature speech | ✅ |
| nsf vocoder | ✅ |
| speaker encoder | ✅ |
| GRL for speaker | Coming soon! |
| one shot vits | Coming soon! |
| band extention | Coming soon! |
| audio enhance | Coming soon! |

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
仅需要以以下文件结构将数据集放入dataset_raw目录即可
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

### 工作目录
    export PYTHONPATH=$PWD

## 数据预处理
- 1， 重采样
    > python svc_resample.py
- 2， 提取音高
    > python prepare/preprocess_f0.py -w data_svc/waves/ -p data_svc/pitch
- 3， 提取内容编码
    > python prepare/preprocess_ppg.py -w data_svc/waves/ -p data_svc/whisper
- 4， 提取音色编码
    > python prepare/preprocess_speaker.py data_svc/waves/ data_svc/speaker
- 5， 提取线性谱
    > python prepare/preprocess_spec.py -w data_svc/waves/ -s data_svc/specs
- 6， 生成训练索引
    > python prepare/preprocess_train.py
- 7， 训练文件调试
    > python prepare/preprocess_zzz.py

## 训练

> python svc_trainer.py -c configs/base.yaml -n sovits5.0


## 推理

> python svc_export.py --config config/base.yaml --checkpoint_path chkpt/sovits5.0/***.pt

> python whisper/inference.py -w test.wav -p temp.ppg.npy

> python svc_inference.py --config config/base.yaml --model sovits5.0.pth --spk ./data_svc/speaker.npy --statics ./data_svc/pitch_statics.npy --wave test.wav

## 数据集
| Name | URL |
| --- | --- |
|KiSing       |http://shijt.site/index.php/2021/05/16/kising-the-first-open-source-mandarin-singing-voice-synthesis-corpus/|
|PopCS 		    |https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/apply_form.md|
|opencpop 	  |https://wenet.org.cn/opencpop/download/|
|Multi-Singer |https://github.com/Multi-Singer/Multi-Singer.github.io|
|M4Singer	    |https://github.com/M4Singer/M4Singer/blob/master/apply_form.md|
|CSD 		      |https://zenodo.org/record/4785016#.YxqrTbaOMU4|
|KSS		      |https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset|
|JVS MuSic	  |https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music|
|PJS		      |https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus|
|JUST Song	  |https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song|
|MUSDB18		  |https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems|
|DSD100 		  |https://sigsep.github.io/datasets/dsd100.html|
|Aishell-3 	  |http://www.aishelltech.com/aishell_3|
|VCTK 		    |https://datashare.ed.ac.uk/handle/10283/2651|

## 代码来源和参考文献

https://github.com/facebookresearch/speech-resynthesis [paper](https://arxiv.org/abs/2104.00355)

https://github.com/jaywalnut310/vits [paper](https://arxiv.org/abs/2106.06103)

https://github.com/openai/whisper/ [paper](https://arxiv.org/abs/2212.04356)

https://github.com/NVIDIA/BigVGAN [paper](https://arxiv.org/abs/2206.04658)

https://github.com/mindslab-ai/univnet [[paper]](https://arxiv.org/abs/2106.07889)

https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

[VI-SVC, Singing Voice Conversion based on VITS decoder](https://zhuanlan.zhihu.com/p/564060769)

[Adapter-Based Extension of Multi-Speaker Text-to-Speech Model for New Speakers](https://arxiv.org/abs/2211.00585)

[AdaSpeech: Adaptive Text to Speech for Custom Voice](https://arxiv.org/pdf/2103.00993.pdf)

## 贡献者

<a href="https://github.com/PlayVoice/so-vits-svc/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PlayVoice/so-vits-svc" />
</a>
