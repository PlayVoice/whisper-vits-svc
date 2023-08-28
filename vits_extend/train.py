import os
import time
import logging
import torch
import numpy as np

from tqdm import tqdm
from vits_extend.dataloader import create_dataloader_train
from vits_extend.dataloader import create_dataloader_eval
from vits_extend.writer import MyWriter
from vits_extend.stft import TacotronSTFT
from vits_extend.validation import validate
from vits.models import SynthesizerTrn
from vits.commons import clip_grad_value_


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


def train(args, chkpt_path, hp):
    device = torch.device("cuda")
    model_g = SynthesizerTrn(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size,
        hp)
    model_g.to(device)
    # train post only
    model_g.train_plug()

    optim_g = torch.optim.AdamW(model_g.parameters(),
                                lr=hp.train.learning_rate,
                                betas=hp.train.betas,
                                eps=hp.train.eps)

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

    step = 0
    init_epoch = 1

    if os.path.isfile(hp.train.pretrain):
        logger.info("Start from 32k pretrain model: %s" % hp.train.pretrain)
        checkpoint = torch.load(hp.train.pretrain, map_location='cpu')
        load_model(model_g, checkpoint['model_g'])

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path, map_location='cpu')
        load_model(model_g, checkpoint['model_g'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        init_epoch = checkpoint['epoch']
        step = checkpoint['step']
    else:
        logger.info("Starting new training run.")

    trainloader = create_dataloader_train(hp)

    for epoch in range(init_epoch, hp.train.epochs):

        if epoch % hp.log.eval_interval == 0:
            with torch.no_grad():
                validate(hp, model_g, valloader, stft, writer, step, device)

        model_g.train()
        loss_a = []
        with tqdm(trainloader) as progress_bar:
            for batch in progress_bar:
                ppg, ppg_l, vec, pit, spk, \
                    spec, spec_l, audio, audio_l = batch

                ppg = ppg.to(device)
                vec = vec.to(device)
                pit = pit.to(device)
                spk = spk.to(device)
                spec = spec.to(device)
                ppg_l = ppg_l.to(device)
                spec_l = spec_l.to(device)

                # generator
                optim_g.zero_grad()
                loss_g = model_g(ppg, vec, pit, spec, spk, ppg_l, spec_l)
                loss_g.backward()
                clip_grad_value_(model_g.parameters(),  None)
                optim_g.step()

                msg = f'Epoch: {epoch}, step: {step} | ' 
                msg = msg + f'diff_loss: {loss_g.item():.3f}'
                progress_bar.set_description(msg)

                step += 1
                # logging
                loss_g = loss_g.item()
                loss_a.append(loss_g)
                writer.log_training(loss_g, step)
                
        logger.info("epoch %d | g %.04f | step %d" % (epoch, np.mean(loss_a), step))

        if epoch % hp.log.save_interval == 0:
            save_path = os.path.join(pth_dir, '%s_%04d.pt'
                                     % (args.name, epoch))
            torch.save({
                'model_g': model_g.state_dict(),
                'optim_g': optim_g.state_dict(),
                'step': step,
                'epoch': epoch,
            }, save_path)
            logger.info("Saved checkpoint to: %s" % save_path)
