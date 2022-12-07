# SoftVC VITS Singing Voice Conversion
## Update
> 断音问题已解决，音质提升了一个档次\
> 2.0版本已经移至 sovits_2.0分支\
> 3.0版本使用FreeVC的代码结构，与旧版本不通用
## 模型简介
歌声音色转换模型，通过SoftVC内容编码器提取源音频语音特征，与F0同时输入VITS替换原本的文本输入达到歌声转换的效果。同时，更换声码器为 [NSF HiFiGAN](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan) 解决断音问题

## 预先下载的模型文件
+ soft vc hubert：[hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)
  + 放在hubert目录下
+ 预训练模型文件 [G_0.pth D_0.pth](https://)
  + 放在logs/48k 目录下
  + 预训练模型为必选项，因为据测试从零开始训练有概率不收敛，同时也能加快训练速度
  + 预训练模型删除了optimizer flow speakerembedding 等无关权重，因此可以认为基本剔除了旧的音色信息
```shell
# 一键下载
# hubert
wget -P hubert/ https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt
# G与D预训练模型
wget -P logs/48k/ https://
wget -P logs/48k/ https://

```


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

## 数据预处理
1. 重采样至 48khz

```shell
python resample.py
 ```
2. 自动划分训练集 验证集 测试集 以及配置文件
```shell
python preprocess_flist_config.py
```
3. 生成hubert与f0
```shell
python preprocess_hubert_f0.py
```
执行完以上步骤后 dataset 目录便是预处理完成的数据，可以删除dataset_raw文件夹了

## 训练
```shell
python train.py -c configs/config.json -m 48k
```

## 推理

使用[inference_main.py](inference_main.py)
+ 更改模型文件为你自己训练的最新模型记录点
+ 将待转化的音频放在raw文件夹下
+ clean_names 写待转化的音频名称
+ trans填写变调半音数量
+ spk_list填写合成的说话人名称

