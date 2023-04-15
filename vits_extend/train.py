import os
import time
import logging
import math
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
import itertools
import traceback

from utils.dataloader import create_dataloader
from utils.writer import MyWriter
from utils.stft import TacotronSTFT
from utils.stft_loss import MultiResolutionSTFTLoss
from model.generator import Generator
from model.discriminator import Discriminator
from .validation import validate


def train(rank, args, chkpt_path, hp, hp_str):

    if args.num_gpus > 1:
        init_process_group(backend=hp.dist_config.dist_backend, init_method=hp.dist_config.dist_url,
                           world_size=hp.dist_config.world_size * args.num_gpus, rank=rank)

    torch.cuda.manual_seed(hp.train.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    model_g = Generator(hp).to(device)
    model_d = Discriminator(hp).to(device)

    optim_g = torch.optim.AdamW(model_g.parameters(),
        lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))
    optim_d = torch.optim.AdamW(model_d.parameters(),
        lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))

    init_epoch = -1
    step = 0

    stft = TacotronSTFT(filter_length=hp.audio.filter_length,
                        hop_length=hp.audio.hop_length,
                        win_length=hp.audio.win_length,
                        n_mel_channels=hp.audio.n_mel_channels,
                        sampling_rate=hp.audio.sampling_rate,
                        mel_fmin=hp.audio.mel_fmin,
                        mel_fmax=hp.audio.mel_fmax,
                        center=False,
                        device=device)
    # define logger, writer, valloader, stft at rank_zero
    if rank == 0:
        pt_dir = os.path.join(hp.log.chkpt_dir, args.name)
        log_dir = os.path.join(hp.log.log_dir, args.name)
        os.makedirs(pt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, '%s-%d.log' % (args.name, time.time()))),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger()
        writer = MyWriter(hp, log_dir)
        valloader = create_dataloader(hp, False)

    if chkpt_path is not None:
        if rank == 0:
            logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path, map_location='cpu')
        model_g.load_state_dict(checkpoint['model_g'])
        model_d.load_state_dict(checkpoint['model_d'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        optim_d.load_state_dict(checkpoint['optim_d'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

        if rank == 0:
            if hp_str != checkpoint['hp_str']:
                logger.warning("New hparams is different from checkpoint. Will use new.")
    else:
        if rank == 0:
            logger.info("Starting new training run.")

    if args.num_gpus > 1:
        model_g = DistributedDataParallel(model_g, device_ids=[rank]).to(device)
        model_d = DistributedDataParallel(model_d, device_ids=[rank]).to(device)

    # this accelerates training when the size of minibatch is always consistent.
    # if not consistent, it'll horribly slow down.
    torch.backends.cudnn.benchmark = True

    trainloader = create_dataloader(hp, True)

    model_g.train()
    model_d.train()

    resolutions = eval(hp.mrd.resolutions)
    stft_criterion = MultiResolutionSTFTLoss(device, resolutions)

    for epoch in itertools.count(init_epoch+1):
        
        if rank == 0 and epoch % hp.log.validation_interval == 0:
            with torch.no_grad():
                validate(hp, args, model_g, model_d, valloader, stft, writer, step, device)

        if rank == 0:
            loader = tqdm.tqdm(trainloader, desc='Loading train data')
        else:
            loader = trainloader

        for spk, ppg, pos, pit, audio in loader:
            spk = spk.to(device)
            ppg = ppg.to(device)
            pos = pos.to(device)
            pit = pit.to(device)
            audio = audio.to(device)

            # generator
            optim_g.zero_grad()
            fake_audio = model_g(spk, ppg, pos, pit)

            # Mel Loss
            mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
            mel_real = stft.mel_spectrogram(audio.squeeze(1))
            mel_loss = F.l1_loss(mel_fake, mel_real) * hp.train.mel_lamb

            # Multi-Resolution STFT Loss
            sc_loss, mag_loss = stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
            stft_loss = (sc_loss + mag_loss) * hp.train.stft_lamb

            res_fake, period_fake = model_d(fake_audio)

            score_loss = 0.0

            for (_, score_fake) in res_fake + period_fake:
                score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))

            score_loss = score_loss / len(res_fake + period_fake)

            # for fast train
            loss_g = score_loss + stft_loss + mel_loss
            # for last train
            # loss_g = score_loss + stft_loss

            loss_g.backward()
            optim_g.step()

            # discriminator

            optim_d.zero_grad()
            res_fake, period_fake = model_d(fake_audio.detach())
            res_real, period_real = model_d(audio)

            loss_d = 0.0
            for (_, score_fake), (_, score_real) in zip(res_fake + period_fake, res_real + period_real):
                loss_d += torch.mean(torch.pow(score_real - 1.0, 2))
                loss_d += torch.mean(torch.pow(score_fake, 2))

            loss_d = loss_d / len(res_fake + period_fake)

            loss_d.backward()
            optim_d.step()

            step += 1
            # logging
            loss_g = loss_g.item()
            loss_d = loss_d.item()
            loss_s = stft_loss.item()
            loss_m = mel_loss.item()

            if rank == 0 and step % hp.log.summary_interval == 0:
                writer.log_training(loss_g, loss_d, loss_m, loss_s, score_loss.item(), step)
                # loader.set_description("g %.04f m %.04f s %.04f d %.04f | step %d" % (loss_g, loss_m, loss_s, loss_d, step))
                logger.info("g %.04f m %.04f s %.04f d %.04f | step %d" % (loss_g, loss_m, loss_s, loss_d, step))

        if rank == 0 and epoch % hp.log.save_interval == 0:
            save_path = os.path.join(pt_dir, '%s_%04d.pt'
                                     % (args.name, epoch))
            torch.save({
                'model_g': (model_g.module if args.num_gpus > 1 else model_g).state_dict(),
                'model_d': (model_d.module if args.num_gpus > 1 else model_d).state_dict(),
                'optim_g': optim_g.state_dict(),
                'optim_d': optim_d.state_dict(),
                'step': step,
                'epoch': epoch,
                'hp_str': hp_str,
            }, save_path)
            logger.info("Saved checkpoint to: %s" % save_path)
