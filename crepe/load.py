import os

import numpy as np
import torch
import crepe
from scipy.io import wavfile


def audio(filename):
    """Load audio from disk"""
    sample_rate, audio = wavfile.read(filename)

    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / np.iinfo(np.int16).max

    # PyTorch is not compatible with non-writeable arrays, so we make a copy
    return torch.tensor(np.copy(audio))[None], sample_rate


def model(device, capacity='full'):
    """Preloads model from disk"""
    # Bind model and capacity
    crepe.infer.capacity = capacity
    crepe.infer.model = crepe.Crepe(capacity)

    # Load weights
    file = os.path.join(os.path.dirname(__file__), 'assets', f'{capacity}.pth')
    crepe.infer.model.load_state_dict(
        torch.load(file, map_location=device))

    # Place on device
    crepe.infer.model = crepe.infer.model.to(torch.device(device))

    # Eval mode
    crepe.infer.model.eval()
