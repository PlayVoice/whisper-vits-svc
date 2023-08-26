from torch.utils.data import DataLoader
from vits.data_utils import TextAudioSpeakerCollate
from vits.data_utils import TextAudioSpeakerSet


def create_dataloader_train(hps):
    collate_fn = TextAudioSpeakerCollate()
    train_dataset = TextAudioSpeakerSet(hps.data.training_files, hps.data)
    train_loader = DataLoader(
        train_dataset,
        num_workers=1,
        shuffle=False,
        batch_size=hps.train.batch_size,
        pin_memory=True,
        collate_fn=collate_fn)
    return train_loader


def create_dataloader_eval(hps):
    collate_fn = TextAudioSpeakerCollate()
    eval_dataset = TextAudioSpeakerSet(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(
        eval_dataset,
        num_workers=1,
        shuffle=False,
        batch_size=hps.train.batch_size,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn)
    return eval_loader
