import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import argparse
from omegaconf import OmegaConf

from vits.models import SynthesizerInfer


def load_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    saved_state_dict = checkpoint_dict["model_g"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    return model


def save_pretrain(checkpoint_path, save_path):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    torch.save({
        'model_g': checkpoint_dict['model_g'],
        'model_d': checkpoint_dict['model_d'],
    }, save_path)


def save_model(model, checkpoint_path):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({'model_g': state_dict}, checkpoint_path)


def main(args):
    hp = OmegaConf.load(args.config)
    model = SynthesizerInfer(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp)

    # save_pretrain(args.checkpoint_path, "sovits5.0.pretrain.pth")
    load_model(args.checkpoint_path, model)
    save_model(model, "sovits5.0.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, required=True,
                        help="path of checkpoint pt file for evaluation")
    args = parser.parse_args()

    main(args)
