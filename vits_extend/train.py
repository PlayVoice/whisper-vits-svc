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

from vits_extend.dataloader import create_dataloader_train
from vits_extend.dataloader import create_dataloader_eval
from vits_extend.writer import MyWriter
from vits_extend.stft import TacotronSTFT
from vits_extend.stft_loss import MultiResolutionSTFTLoss
from vits_extend.validation import validate
from vits_decoder.discriminator import Discriminator
from vits.models import SynthesizerTrn
from vits import commons
from vits.losses import kl_loss
from vits.commons import clip_grad_value_


def load_part(model, saved_state_dict):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('TODO'):
            new_state_dict[k] = v
        else:
            new_state_dict[k] = saved_state_dict[k]
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    return model


def load_model(model, saved_state_dict):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            print("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    return model


def train(rank, args, chkpt_path, hp, hp_str):

    if args.num_gpus > 1:
        init_process_group(backend=hp.dist_config.dist_backend, init_method=hp.dist_config.dist_url,
                           world_size=hp.dist_config.world_size * args.num_gpus, rank=rank)

    torch.cuda.manual_seed(hp.train.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    model_g = SynthesizerTrn(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp).to(device)
    model_d = Discriminator(hp).to(device)

    optim_g = torch.optim.AdamW(model_g.parameters(),
                                lr=hp.train.learning_rate, betas=hp.train.betas, eps=hp.train.eps)
    optim_d = torch.optim.AdamW(model_d.parameters(),
                                lr=hp.train.learning_rate, betas=hp.train.betas, eps=hp.train.eps)

    init_epoch = 1
    step = 0

    stft = TacotronSTFT(filter_length=hp.data.filter_length,
                        hop_length=hp.data.hop_length,
                        win_length=hp.data.win_length,
                        n_mel_channels=hp.data.mel_channels,
                        sampling_rate=hp.data.sampling_rate,
                        mel_fmin=hp.data.mel_fmin,
                        mel_fmax=hp.data.mel_fmax,
                        center=False,
                        device=device)
    # define logger, writer, valloader, stft at rank_zero
    if rank == 0:
        pth_dir = os.path.join(hp.log.pth_dir, args.name)
        log_dir = os.path.join(hp.log.log_dir, args.name)
        os.makedirs(pth_dir, exist_ok=True)
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
        valloader = create_dataloader_eval(hp)

    if os.path.isfile(hp.train.pretrain):
        if rank == 0:
            logger.info("Start from 32k pretrain model: %s" % hp.train.pretrain)
        checkpoint = torch.load(hp.train.pretrain, map_location='cpu')
        load_model(model_g, checkpoint['model_g'])
        load_model(model_d, checkpoint['model_d'])

    if chkpt_path is not None:
        if rank == 0:
            logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path, map_location='cpu')
        load_model(model_g, checkpoint['model_g'])
        load_model(model_d, checkpoint['model_d'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        optim_d.load_state_dict(checkpoint['optim_d'])
        init_epoch = checkpoint['epoch']
        step = checkpoint['step']

        if rank == 0:
            if hp_str != checkpoint['hp_str']:
                logger.warning("New hparams is different from checkpoint. Will use new.")
    else:
        if rank == 0:
            logger.info("Starting new training run.")

    if args.num_gpus > 1:
        model_g = DistributedDataParallel(model_g, device_ids=[rank])
        model_d = DistributedDataParallel(model_d, device_ids=[rank])

    # this accelerates training when the size of minibatch is always consistent.
    # if not consistent, it'll horribly slow down.
    torch.backends.cudnn.benchmark = True

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hp.train.lr_decay, last_epoch=init_epoch-2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hp.train.lr_decay, last_epoch=init_epoch-2)

    stft_criterion = MultiResolutionSTFTLoss(device, eval(hp.mrd.resolutions))
    spkc_criterion = nn.CosineEmbeddingLoss()

    trainloader = create_dataloader_train(hp, args.num_gpus, rank)

    for epoch in range(init_epoch, hp.train.epochs):

        trainloader.batch_sampler.set_epoch(epoch)

        if rank == 0 and epoch % hp.log.eval_interval == 0:
            with torch.no_grad():
                validate(hp, args, model_g, model_d, valloader, stft, writer, step, device)

        if rank == 0:
            loader = tqdm.tqdm(trainloader, desc='Loading train data')
        else:
            loader = trainloader

        model_g.train()
        model_d.train()

        for ppg, ppg_l, pit, spk, spec, spec_l, audio, audio_l in loader:

            ppg = ppg.to(device)
            pit = pit.to(device)
            spk = spk.to(device)
            spec = spec.to(device)
            audio = audio.to(device)
            ppg_l = ppg_l.to(device)
            spec_l = spec_l.to(device)
            audio_l = audio_l.to(device)

            # generator
            optim_g.zero_grad()

            fake_audio, ids_slice, z_mask, \
                (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r), spk_preds = model_g(
                    ppg, pit, spec, spk, ppg_l, spec_l)


            audio = commons.slice_segments(
                audio, ids_slice * hp.data.hop_length, hp.data.segment_size)  # slice
            # Spk Loss
            spk_loss = spkc_criterion(spk, spk_preds, torch.Tensor(spk_preds.size(0))
                                .to(device).fill_(1.0))
            # Mel Loss
            mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
            mel_real = stft.mel_spectrogram(audio.squeeze(1))
            mel_loss = F.l1_loss(mel_fake, mel_real) * hp.train.c_mel

            # Multi-Resolution STFT Loss
            sc_loss, mag_loss = stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
            stft_loss = (sc_loss + mag_loss) * hp.train.c_stft

            # Generator Loss
            disc_fake = model_d(fake_audio)
            score_loss = 0.0
            for (_, score_fake) in disc_fake:
                score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))
            score_loss = score_loss / len(disc_fake)

            # Feature Loss
            disc_real = model_d(audio)
            feat_loss = 0.0
            for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                for fake, real in zip(feat_fake, feat_real):
                    feat_loss += torch.mean(torch.abs(fake - real))
            feat_loss = feat_loss / len(disc_fake)
            feat_loss = feat_loss * 2

            # Kl Loss
            loss_kl_f = kl_loss(z_f, logs_q, m_p, logs_p, logdet_f, z_mask) * hp.train.c_kl
            loss_kl_r = kl_loss(z_r, logs_p, m_q, logs_q, logdet_r, z_mask) * hp.train.c_kl

            # Loss
            loss_g = score_loss + feat_loss + mel_loss + stft_loss + loss_kl_f + loss_kl_r * 0.5 + spk_loss * 2
            loss_g.backward()
            clip_grad_value_(model_g.parameters(),  None)
            optim_g.step()

            # discriminator
            optim_d.zero_grad()
            disc_fake = model_d(fake_audio.detach())
            disc_real = model_d(audio)

            loss_d = 0.0
            for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
                loss_d += torch.mean(torch.pow(score_real - 1.0, 2))
                loss_d += torch.mean(torch.pow(score_fake, 2))
            loss_d = loss_d / len(disc_fake)

            loss_d.backward()
            clip_grad_value_(model_d.parameters(),  None)
            optim_d.step()

            step += 1
            # logging
            loss_g = loss_g.item()
            loss_d = loss_d.item()
            loss_s = stft_loss.item()
            loss_m = mel_loss.item()
            loss_k = loss_kl_f.item()
            loss_r = loss_kl_r.item()
            loss_i = spk_loss.item()

            if rank == 0 and step % hp.log.info_interval == 0:
                writer.log_training(
                    loss_g, loss_d, loss_m, loss_s, loss_k, loss_r, score_loss.item(), step)
                logger.info("g %.04f m %.04f s %.04f d %.04f k %.04f r %.04f i %.04f | step %d" % (
                    loss_g, loss_m, loss_s, loss_d, loss_k, loss_r, loss_i, step))

        if rank == 0 and epoch % hp.log.save_interval == 0:
            save_path = os.path.join(pth_dir, '%s_%04d.pt'
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

        scheduler_g.step()
        scheduler_d.step()
