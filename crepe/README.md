<h1 align="center">torchcrepe</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/torchcrepe.svg)](https://pypi.python.org/pypi/torchcrepe)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/torchcrepe)](https://pepy.tech/project/torchcrepe)

</div>

Pytorch implementation of the CREPE [1] pitch tracker. The original Tensorflow
implementation can be found [here](https://github.com/marl/crepe/). The
provided model weights were obtained by converting the "tiny" and "full" models
using [MMdnn](https://github.com/microsoft/MMdnn), an open-source model
management framework.


## Installation
Perform the system-dependent PyTorch install using the instructions found
[here](https://pytorch.org/).

`pip install torchcrepe`


## Usage

### Computing pitch and periodicity from audio


```python
import torchcrepe


# Load audio
audio, sr = torchcrepe.load.audio( ... )

# Here we'll use a 5 millisecond hop length
hop_length = int(sr / 200.)

# Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
# This would be a reasonable range for speech
fmin = 50
fmax = 550

# Select a model capacity--one of "tiny" or "full"
model = 'tiny'

# Choose a device to use for inference
device = 'cuda:0'

# Pick a batch size that doesn't cause memory errors on your gpu
batch_size = 2048

# Compute pitch using first gpu
pitch = torchcrepe.predict(audio,
                           sr,
                           hop_length,
                           fmin,
                           fmax,
                           model,
                           batch_size=batch_size,
                           device=device)
```

A periodicity metric similar to the Crepe confidence score can also be
extracted by passing `return_periodicity=True` to `torchcrepe.predict`.


### Decoding

By default, `torchcrepe` uses Viterbi decoding on the softmax of the network
output. This is different than the original implementation, which uses a
weighted average near the argmax of binary cross-entropy probabilities.
The argmax operation can cause double/half frequency errors. These can be
removed by penalizing large pitch jumps via Viterbi decoding. The `decode`
submodule provides some options for decoding.

```python
# Decode using viterbi decoding (default)
torchcrepe.predict(..., decoder=torchcrepe.decode.viterbi)

# Decode using weighted argmax (as in the original implementation)
torchcrepe.predict(..., decoder=torchcrepe.decode.weighted_argmax)

# Decode using argmax
torchcrepe.predict(..., decoder=torchcrepe.decode.argmax)
```


### Filtering and thresholding

When periodicity is low, the pitch is less reliable. For some problems, it
makes sense to mask these less reliable pitch values. However, the periodicity
can be noisy and the pitch has quantization artifacts. `torchcrepe` provides
submodules `filter` and `threshold` for this purpose. The filter and threshold
parameters should be tuned to your data. For clean speech, a 10-20 millisecond
window with a threshold of 0.21 has worked.

```python
# We'll use a 15 millisecond window assuming a hop length of 5 milliseconds
win_length = 3

# Median filter noisy confidence value
periodicity = torchcrepe.filter.median(periodicity, win_length)

# Remove inharmonic regions
pitch = torchcrepe.threshold.At(.21)(pitch, periodicity)

# Optionally smooth pitch to remove quantization artifacts
pitch = torchcrepe.filter.mean(pitch, win_length)
```

For more fine-grained control over pitch thresholding, see
`torchcrepe.threshold.Hysteresis`. This is especially useful for removing
spurious voiced regions caused by noise in the periodicity values, but
has more parameters and may require more manual tuning to your data.

CREPE was not trained on silent audio. Therefore, it sometimes assigns high
confidence to pitch bins in silent regions. You can use
`torchcrepe.threshold.Silence` to manually set the periodicity in silent
regions to zero.

```python
periodicity = torchcrepe.threshold.Silence(-60.)(periodicity,
                                                 audio,
                                                 sr,
                                                 hop_length)
```


### Computing the CREPE model output activations

```python
batch = next(torchcrepe.preprocess(audio, sr, hop_length))
probabilities = torchcrepe.infer(batch)
```


### Computing the CREPE embedding space

As in Differentiable Digital Signal Processing [2], this uses the output of the
fifth max-pooling layer as a pretrained pitch embedding

```python
embeddings = torchcrepe.embed(audio, sr, hop_length)
```

### Computing from files

`torchcrepe` defines the following functions convenient for predicting
directly from audio files on disk. Each of these functions also takes
a `device` argument that can be used for device placement (e.g.,
`device='cuda:0'`).

```python
torchcrepe.predict_from_file(audio_file, ...)
torchcrepe.predict_from_file_to_file(
    audio_file, output_pitch_file, output_periodicity_file, ...)
torchcrepe.predict_from_files_to_files(
    audio_files, output_pitch_files, output_periodicity_files, ...)

torchcrepe.embed_from_file(audio_file, ...)
torchcrepe.embed_from_file_to_file(audio_file, output_file, ...)
torchcrepe.embed_from_files_to_files(audio_files, output_files, ...)
```

### Command-line interface

```bash
usage: python -m torchcrepe
    [-h]
    --audio_files AUDIO_FILES [AUDIO_FILES ...]
    --output_files OUTPUT_FILES [OUTPUT_FILES ...]
    [--hop_length HOP_LENGTH]
    [--output_periodicity_files OUTPUT_PERIODICITY_FILES [OUTPUT_PERIODICITY_FILES ...]]
    [--embed]
    [--fmin FMIN]
    [--fmax FMAX]
    [--model MODEL]
    [--decoder DECODER]
    [--gpu GPU]
    [--no_pad]

optional arguments:
  -h, --help            show this help message and exit
  --audio_files AUDIO_FILES [AUDIO_FILES ...]
                        The audio file to process
  --output_files OUTPUT_FILES [OUTPUT_FILES ...]
                        The file to save pitch or embedding
  --hop_length HOP_LENGTH
                        The hop length of the analysis window
  --output_periodicity_files OUTPUT_PERIODICITY_FILES [OUTPUT_PERIODICITY_FILES ...]
                        The file to save periodicity
  --embed               Performs embedding instead of pitch prediction
  --fmin FMIN           The minimum frequency allowed
  --fmax FMAX           The maximum frequency allowed
  --model MODEL         The model capacity. One of "tiny" or "full"
  --decoder DECODER     The decoder to use. One of "argmax", "viterbi", or
                        "weighted_argmax"
  --gpu GPU             The gpu to perform inference on
  --no_pad              Whether to pad the audio
```


## Tests

The module tests can be run as follows.

```bash
pip install pytest
pytest
```


## References
[1] J. W. Kim, J. Salamon, P. Li, and J. P. Bello, “Crepe: A
Convolutional Representation for Pitch Estimation,” in 2018 IEEE
International Conference on Acoustics, Speech and Signal
Processing (ICASSP).

[2] J. H. Engel, L. Hantrakul, C. Gu, and A. Roberts,
“DDSP: Differentiable Digital Signal Processing,” in
2020 International Conference on Learning
Representations (ICLR).
