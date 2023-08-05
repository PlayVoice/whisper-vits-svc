[![Python Package using Conda](https://github.com/groupmm/libf0/actions/workflows/test_conda.yml/badge.svg)](https://github.com/groupmm/libf0/actions/workflows/test_conda.yml)
[![Python package](https://github.com/groupmm/libf0/actions/workflows/test_pip.yml/badge.svg)](https://github.com/groupmm/libf0/actions/workflows/test_pip.yml)


# libf0

This repository contains a Python package called libf0 which provides open-source  implementations for four popular model-based F0-estimation approaches, YIN (Cheveigné & Kawahara, 2002), pYIN (Mauch & Dixon, 2014), an approach inspired by Melodia (Salamon & Gómez, 2012), and SWIPE (Camacho & Harris, 2008).

If you use the libf0 in your research, please consider the following references.

## References

Sebastian Rosenzweig, Simon Schwär, and Meinard Müller.
[libf0: A Python Library for Fundamental Frequency Estimation.](https://archives.ismir.net/ismir2022/latebreaking/000003.pdf)
In Late Breaking Demos of the International Society for Music Information Retrieval Conference (ISMIR), Bengaluru, India, 2022.

Alain de Cheveigné and Hideki Kawahara.
YIN, a fundamental frequency estimator for speech and music. Journal of the Acoustical Society of America (JASA), 111(4):1917–1930, 2002.

Matthias Mauch and Simon Dixon.
pYIN: A fundamental frequency estimator using probabilistic threshold distributions. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 659–663, Florence, Italy, 2014.

Justin Salamon and Emilia Gómez.
Melody extraction from polyphonic music signals using pitch contour characteristics. IEEE Transactions on Audio, Speech, and Language Processing, 20(6):
1759–1770, 2012.

Arturo Camacho and John G. Harris.
A sawtooth waveform inspired pitch estimator for speech and music. The Journal of the Acoustical Society of America, 124(3):1638–1652, 2008.

Meinard Müller. Fundamentals of Music Processing – Using Python and Jupyter Notebooks. Springer Verlag, 2nd edition, 2021. ISBN 978-3-030-69807-2. doi: 10.1007/978-3-030-69808-9.


## Installing

If you just want to try our example notebook, you can run it using Binder directly in your browser: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/groupmm/libf0/HEAD)

To install the libf0 locally, you can use the Python package manager pip:

```
pip install libf0
```

We recommend to do this inside a conda or virtual environment (requiring at least Python 3.7).
If you want to run the example notebook locally, you **must** first install libf0 to resolve all dependencies. Then, you can clone this repository using

```
git clone https://github.com/groupmm/libf0.git
```
install Jupyter using

```
pip install jupyter
```

and then start the notebook server via

```
jupyter notebook
```


## Documentation
There is also an API documentation for libf0:

https://groupmm.github.io/libf0

## Contributing

We are happy for suggestions and contributions. We would be grateful for either directly contacting us via email (meinard.mueller@audiolabs-erlangen.de) or for creating an issue in our Github repository. Please do not submit a pull request without prior consultation with us.

## Tests

We provide automated tests for each algorithm. To execute the test script, you will need to install extra requirements for testing:

```
pip install 'libf0[tests]'
pytest tests
```

## Licence

The code for this toolbox is published under an MIT licence.

## Acknowledgements

This work was supported by the German Research Foundation (MU 2686/13-1, SCHE 280/20-1). We thank Edgar Suárez and Vojtěch Pešek for helping with the implementations. Furthermore, we thank Fatemeh Eftekhar and Maryam Pirmoradi for testing the toolbox. The International Audio Laboratories Erlangen are a joint institution of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) and Fraunhofer Institute for Integrated Circuits IIS.
