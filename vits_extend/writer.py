from torch.utils.tensorboard import SummaryWriter
import numpy as np
import librosa

from .plotting import plot_waveform_to_numpy, plot_spectrogram_to_numpy

class MyWriter(SummaryWriter):
    def __init__(self, hp, logdir):
        super(MyWriter, self).__init__(logdir)
        self.sample_rate = hp.audio.sampling_rate
        self.is_first = True

    def log_training(self, g_loss, d_loss, mel_loss, stft_loss, score_loss, step):
        self.add_scalar('train/g_loss', g_loss, step)
        self.add_scalar('train/d_loss', d_loss, step)
        
        self.add_scalar('train/score_loss', score_loss, step)
        self.add_scalar('train/stft_loss', stft_loss, step)
        self.add_scalar('train/mel_loss', mel_loss, step)

    def log_validation(self, mel_loss, generator, discriminator, step):
        self.add_scalar('validation/mel_loss', mel_loss, step)

        self.log_histogram(generator, step)
        self.log_histogram(discriminator, step)
        if self.is_first:
            self.is_first = False

    def log_fig_audio(self, target, prediction, spec_fake, spec_real, idx, step):
        if idx == 0:
            spec_fake = librosa.amplitude_to_db(spec_fake, ref=np.max,top_db=80.)
            spec_real = librosa.amplitude_to_db(spec_real, ref=np.max,top_db=80.)
            self.add_image('spec/predicted', plot_spectrogram_to_numpy(spec_fake), step)
            self.add_image('spec/error', plot_spectrogram_to_numpy(np.power(spec_real - spec_fake, 2)), step)
            self.add_image('waveform/predicted', plot_waveform_to_numpy(prediction), step)
            if self.is_first:
                self.add_image('spec/target', plot_spectrogram_to_numpy(spec_real), step)
                self.add_image('waveform/target', plot_waveform_to_numpy(target), step)

        self.add_audio('predicted/raw_audio_%d' % idx, prediction, step, self.sample_rate)
        if self.is_first:
            self.add_audio('target/raw_audio_%d' % idx, target, step, self.sample_rate)


    def log_histogram(self, model, step):
        for tag, value in model.named_parameters():
            self.add_histogram(tag.replace('.', '/'), value.cpu().detach().numpy(), step)
