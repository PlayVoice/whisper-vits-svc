<div align="center">
<h1> Plug-in diffusion based on Grad-TTS from HUAWEI Noah's Ark Lab </h1>

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/maxmax20160403/sovits5.0)
<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/PlayVoice/so-vits-svc-5.0">
<img alt="GitHub forks" src="https://img.shields.io/github/forks/PlayVoice/so-vits-svc-5.0">
<img alt="GitHub issues" src="https://img.shields.io/github/issues/PlayVoice/so-vits-svc-5.0">
<img alt="GitHub" src="https://img.shields.io/github/license/PlayVoice/so-vits-svc-5.0">

纯粹的算法研究分支~~~

![vits-5.0-frame](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/3854b281-8f97-4016-875b-6eb663c92466)
原始框架

![plug-in-diffusion](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/54a61c90-a97b-404d-9cc9-a2151b2db28f)
Diffusion插件

![svc_plug_in](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/aa209a3e-4e6f-4e0c-833b-345d5e757e8e)
</div>

## 说明
- 因为本项目本来电音就少，所以看不出什么效果
- 仅供算法研究

## 训练
- 完成`bigvgan-mix-v2`主模型的训练
- 创建工作路径，拉取分支代码：与`bigvgan-mix-v2`代码不同
- 安装额外依赖`pip install einops`
- 拷贝`bigvgan-mix-v2`的训练数据`data_svc`与`files`到当前工作目录：与`bigvgan-mix-v2`训练数据一样
- 在`configs/base.yaml`中指定主模型路径，如`pretrain: "bigvgan-mix-v2/chkpt/sovits5.0/sovits5.0_0500.pt"`
- 启动训练`python svc_trainer.py --config configs/base.yaml --name plug`
- 训练速度非常快~

## 推理
- 推理方法与原模型一致
- `svc_inference.py`代码有微小修改

## 参考
https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS [paper](https://arxiv.org/abs/2105.06337)
