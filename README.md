# SoftVC VITS Singing Voice Conversion

## 模型简介
歌声音色转换模型，通过SoftVC内容编码器提取源音频语音特征，与F0同时输入VITS替换原本的文本输入达到歌声转换的效果。同时，更换声码器为 [NSF HiFiGAN](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan) 解决断音问题
## 注意
+ 当前分支是32khz版本的分支，32khz模型推理更快，显存占用大幅减小，数据集所占硬盘空间也大幅降低，推荐训练该版本模型
+ 如果是使用git clone 下载的仓库需要先 git checkout 32k 切换至32k分支
+ 如果之前训练了48khz的模型想切换到32khz, 则需要重新执行完整的预处理过程（重采样、生成配置文件、生成f0）
但是可以直接加载旧的48khz模型进行加训
## 预先下载的模型文件
+ soft vc hubert：[hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)
  + 放在hubert目录下
+ 预训练底模文件 [G_0.pth](https://huggingface.co/innnky/sovits_pretrained/resolve/main/G_0.pth) 与 [D_0.pth](https://huggingface.co/innnky/sovits_pretrained/resolve/main/D_0.pth)
  + 放在logs/32k 目录下
  + 预训练底模为必选项，因为据测试从零开始训练有概率不收敛，同时底模也能加快训练速度
  + 预训练底模训练数据集包含云灏 即霜 辉宇·星AI 派蒙 绫地宁宁，覆盖男女生常见音域，可以认为是相对通用的底模
  + 底模删除了optimizer speaker_embedding 等无关权重, 只可以用于初始化训练，无法用于推理
  + 该底模和48khz底模通用
```shell
# 一键下载
# hubert
wget -P hubert/ https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt
# G与D预训练模型
wget -P logs/32k/ https://huggingface.co/innnky/sovits_pretrained/resolve/main/G_0.pth
wget -P logs/32k/ https://huggingface.co/innnky/sovits_pretrained/resolve/main/D_0.pth

```
## colab一键数据集制作、训练脚本
[一键colab](https://colab.research.google.com/drive/1rCUOOVG7-XQlVZuWRAj5IpGrMM8t07pE?usp=sharing)
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
1. 重采样至 32khz

```shell
python resample.py
 ```
2. 自动划分训练集 验证集 测试集 以及自动生成配置文件
```shell
python preprocess_flist_config.py
# 注意
# 自动生成的配置文件中，说话人数量n_speakers会自动按照数据集中的人数而定
# 为了给之后添加说话人留下一定空间，n_speakers自动设置为 当前数据集人数乘2
# 如果想多留一些空位可以在此步骤后 自行修改生成的config.json中n_speakers数量
# 一旦模型开始训练后此项不可再更改
```
3. 生成hubert与f0
```shell
python preprocess_hubert_f0.py
```
执行完以上步骤后 dataset 目录便是预处理完成的数据，可以删除dataset_raw文件夹了

## 训练
```shell
python train.py -c configs/config.json -m 32k
```

## 训练中追加说话人数据
基本类似预处理过程
1. 将新追加的说话人数据按之前的结构放入dataset_raw目录下，并重采样至32khz
```shell
python resample.py
 ```
2. 使用`add_speaker.py`重新生成训练集、验证集，重新生成配置文件
```shell
python add_speaker.py
```
3. 重新生成hubert与f0
```shell
python preprocess_hubert_f0.py
```
之后便可以删除dataset_raw文件夹了

## 推理

使用[inference_main.py](inference_main.py)
+ 更改model_path为你自己训练的最新模型记录点
+ 将待转换的音频放在raw文件夹下
+ clean_names 写待转换的音频名称
+ trans 填写变调半音数量
+ spk_list 填写合成的说话人名称

