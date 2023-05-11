import argparse
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf


torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model for logging, saving checkpoint")
    parser.add_argument('-e', '--extend', type=int, default=0,
                        help="extend trian")
    args = parser.parse_args()

    if (args.extend > 0):
        from vits_extend.trainEx import train
    else:
        from vits_extend.train import train

    hp = OmegaConf.load(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    assert hp.data.hop_length == 320, \
        'hp.data.hop_length must be equal to 320, got %d' % hp.data.hop_length

    args.num_gpus = 0
    torch.manual_seed(hp.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.train.seed)
        args.num_gpus = torch.cuda.device_count()
        print('Batch size per GPU :', hp.train.batch_size)

        if args.num_gpus > 1:
            mp.spawn(train, nprocs=args.num_gpus,
                     args=(args, args.checkpoint_path, hp, hp_str,))
        else:
            train(0, args, args.checkpoint_path, hp, hp_str)
    else:
        print('No GPU find!')
