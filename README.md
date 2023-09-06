<div align="center">
<h1> Plug-in diffusion based on Grad-TTS from HUAWEI Noah's Ark Lab </h1>

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/maxmax20160403/sovits5.0)
<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/PlayVoice/so-vits-svc-5.0">
<img alt="GitHub forks" src="https://img.shields.io/github/forks/PlayVoice/so-vits-svc-5.0">
<img alt="GitHub issues" src="https://img.shields.io/github/issues/PlayVoice/so-vits-svc-5.0">
<img alt="GitHub" src="https://img.shields.io/github/license/PlayVoice/so-vits-svc-5.0">

This branch is for AI developers ~~~

![vits-5.0-frame](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/3854b281-8f97-4016-875b-6eb663c92466)
Base framework ~~~

![plug-in-diffusion](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/54a61c90-a97b-404d-9cc9-a2151b2db28f)
Plug-In-Diffusion

![svc_plug_in](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/aa209a3e-4e6f-4e0c-833b-345d5e757e8e)
</div>

## Notices
- It looks like it's useless, but it seems to be somewhat useful
- 好像没啥用，好像有点用

## 训练
1. Complete the training of the `bigvgan-mix-v2` master model
    
    完成 `bigvgan-mix-v2` 主模型的训练

2. Create a working path and pull the branch codes: different from the `bigvgan-mix-v2`
    
    创建工作路径，拉取分支代码：与 `bigvgan-mix-v2` 不同

3. install additional dependencies for `diffusion`:
    
    为 `diffusion` 安装额外依赖: 
    
    > `pip install einops`

4. Copy `bigvgan-mix-v2` training data `data_svc` and `files` to the current working directory: same as `bigvgan-mix-v2` training data
    
    拷贝 `bigvgan-mix-v2` 的训练数据 `data_svc` 与 `files` 到当前工作目录：与 `bigvgan-mix-v2` 训练数据一样

5. Specify the master model path in `configs/base.yaml`:

    在 `configs/base.yaml` 中指定主模型路径: 
    
    > `pretrain: "bigvgan-mix-v2/chkpt/sovits5.0/sovits5.0_0500.pt"`

6. Start train
    
    启动训练

    > `python svc_trainer.py --config configs/base.yaml --name plug`

    Check the log to be sure: your master model is loaded
    ```
    python svc_trainer.py --config configs/base.yaml --name plug
    Batch size per GPU : 8
    ----------10----------
    2023-09-06 06:31:23,136 - INFO - Start from 32k pretrain model: sovits5.0_1100. pt
    plug.estimator.spk_mlp.0.weight is not in the checkpoint
    plug.estimator.spk_mlp.0.bias is not in the checkpoint
    plug.estimator.spk_mlp.2.weight is not in the checkpoint
    plug.estimator.spk_mlp.2.bias is not in the checkpoint
    plug.estimator.mlp.0.weight is not in the checkpoint
    plug.estimator.mlp.0.bias is not in the checkpoint
    plug.estimator.mlp.2.weight is not in the checkpoint
    plug.estimator.mlp.2.bias is not in the checkpoint
    plug.estimator.downs.0.0.mlp.1.weight is not in the checkpoint
    plug.estimator.downs.0.0.mlp.1.bias is not in the checkpoint
    plug.estimator.downs.0.0.block1.block.0.weight is not in the checkpoint
    plug.estimator.downs.0.0.block1.block.0.bias is not in the checkpoint
    ```

## Inference

> `python svc_inference.py --config configs/base.yaml --model chkpt/plug/plug_***.pt --spk ./data_svc/singer/your_singer.spk.npy --wave test.wav`

`svc_inference.py` has a small changes from `bigvgan-mix-v2`

## Reference
https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS
