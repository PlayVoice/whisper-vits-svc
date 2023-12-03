import abc
import logging

import torch

from feature_retrieval import FaissRetrievableFeatureIndex

logger = logging.getLogger(__name__)


class IRetrieval(abc.ABC):
    @abc.abstractmethod
    def retriv_whisper(self, vec: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def retriv_hubert(self, vec: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DummyRetrieval(IRetrieval):
    def retriv_whisper(self, vec: torch.FloatTensor) -> torch.FloatTensor:
        logger.debug("start dummy retriv whisper")
        return vec.clone().to(torch.device("cpu"))

    def retriv_hubert(self, vec: torch.FloatTensor) -> torch.FloatTensor:
        logger.debug("start dummy retriv hubert")
        return vec.clone().to(torch.device("cpu"))


class FaissIndexRetrieval(IRetrieval):
    def __init__(self, hubert_index: FaissRetrievableFeatureIndex, whisper_index: FaissRetrievableFeatureIndex) -> None:
        self._hubert_index = hubert_index
        self._whisper_index = whisper_index

    def retriv_whisper(self, vec: torch.Tensor) -> torch.Tensor:
        logger.debug("start retriv whisper")
        np_vec = self._whisper_index.retriv(vec.numpy())
        return torch.from_numpy(np_vec)

    def retriv_hubert(self, vec: torch.Tensor) -> torch.Tensor:
        logger.debug("start retriv hubert")
        np_vec = self._hubert_index.retriv(vec.numpy())
        return torch.from_numpy(np_vec)
