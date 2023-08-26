import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
from omegaconf import OmegaConf
from vits_extend.train import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model for logging, saving checkpoint")
    args = parser.parse_args()

    hp = OmegaConf.load(args.config)
    assert hp.data.hop_length == 320, \
        'hp.data.hop_length must be equal to 320, got %d' % hp.data.hop_length

    torch.manual_seed(hp.train.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(hp.train.seed)
        args.num_gpus = torch.cuda.device_count()
        print('Batch size per GPU :', hp.train.batch_size)
        train(args, args.checkpoint_path, hp)
    else:
        print('No GPU find!')
