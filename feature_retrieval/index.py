import abc
import logging
import math
import time
from pathlib import Path
from typing import TypeVar, Generic, cast, Any

import numpy as np
import numpy.typing as npt

from tqdm import tqdm

import faiss
from faiss import IndexIVF, Index

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Index)
NumpyArray = npt.NDArray[np.float32]


class FaissFeatureIndex(Generic[T], abc.ABC):
    def __init__(self, index: T) -> None:
        self._index = index

    def save(self, filepath: Path, rewrite: bool = False) -> None:
        if filepath.exists() and not rewrite:
            raise FileExistsError(f"index already exists by path {filepath}")
        faiss.write_index(self._index, str(filepath))


class FaissRetrievableFeatureIndex(FaissFeatureIndex[Index], abc.ABC):
    """retrieve voice feature vectors by faiss index"""

    def __init__(self, index: T, ratio: float, n_nearest_vectors: int) -> None:
        super().__init__(index=index)
        if index.metric_type != self.supported_distance:
            raise ValueError(f"index metric type {index.metric_type=} is unsupported {self.supported_distance=}")

        if 1 > n_nearest_vectors:
            raise ValueError("n-retrieval-vectors must be gte 1")
        self._n_nearest = n_nearest_vectors

        if 0 > ratio > 1:
            raise ValueError(f"{ratio=} must be in rage (0, 1)")
        self._ratio = ratio

    @property
    @abc.abstractmethod
    def supported_distance(self) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def _weight_nearest_vectors(self, nearest_vectors: NumpyArray, scores: NumpyArray) -> NumpyArray:
        raise NotImplementedError

    def retriv(self, features: NumpyArray) -> NumpyArray:
        # use method search_and_reconstruct instead of recreating the whole matrix
        scores, _, nearest_vectors = self._index.search_and_reconstruct(features, k=self._n_nearest)
        weighted_nearest_vectors = self._weight_nearest_vectors(nearest_vectors, scores)
        retriv_vector = (1 - self._ratio) * features + self._ratio * weighted_nearest_vectors
        return retriv_vector


class FaissRVCRetrievableFeatureIndex(FaissRetrievableFeatureIndex):
    """
    retrieve voice encoded features with algorith from RVC repository
    https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
    """

    @property
    def supported_distance(self) -> Any:
        return faiss.METRIC_L2

    def _weight_nearest_vectors(self, nearest_vectors: NumpyArray, scores: NumpyArray) -> NumpyArray:
        """
        magic code from original RVC
        https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/86ed98aacaa8b2037aad795abd11cdca122cf39f/vc_infer_pipeline.py#L213C18-L213C19

        nearest_vectors dim (n_nearest, vector_dim)
        scores dim (num_vectors, n_nearest)
        """
        logger.debug("shape: nv=%s sc=%s", nearest_vectors.shape, scores.shape)
        weight = np.square(1 / scores)
        weight /= weight.sum(axis=1, keepdims=True)
        weight = np.expand_dims(weight, axis=2)
        weighted_nearest_vectors = np.sum(nearest_vectors * weight, axis=1)
        logger.debug(
            "shape: nv=%s weight=%s weight_nearest=%s",
            nearest_vectors.shape,
            weight.shape,
            weighted_nearest_vectors.shape,
        )
        return cast(NumpyArray, weighted_nearest_vectors)


class FaissIVFTrainableFeatureIndex(FaissFeatureIndex[IndexIVF]):
    """IVF faiss index that can train and add feature vectors"""

    def __init__(self, index: IndexIVF, batch_size: int) -> None:
        super().__init__(index=index)
        self._batch_size = batch_size

    @property
    def _trained_index(self) -> IndexIVF:
        if not self._index.is_trained:
            raise RuntimeError("index needs to be trained first")
        return self._index

    @property
    def _not_trained_index(self) -> IndexIVF:
        if self._index.is_trained:
            raise RuntimeError("index is already trained")
        return self._index

    def _batch_count(self, feature_matrix: NumpyArray) -> int:
        return math.ceil(feature_matrix.shape[0] / self._batch_size)

    def _split_matrix_by_batch(self, feature_matrix: NumpyArray) -> list[NumpyArray]:
        return np.array_split(feature_matrix, indices_or_sections=self._batch_count(feature_matrix), axis=0)

    def _train_index(self, train_feature_matrix: NumpyArray) -> None:
        start = time.monotonic()
        self._not_trained_index.train(train_feature_matrix)
        took = time.monotonic() - start
        logger.info("index is trained. Took %.2f seconds", took)

    def add_to_index(self, feature_matrix: NumpyArray) -> None:
        n_batches = self._batch_count(feature_matrix)
        logger.info("adding %s batches to index", n_batches)
        start = time.monotonic()
        for batch in tqdm(self._split_matrix_by_batch(feature_matrix), total=n_batches):
            self._trained_index.add(batch)
        took = time.monotonic() - start
        logger.info("all batches added. Took %.2f seconds", took)

    def add_with_train(self, feature_matrix: NumpyArray) -> None:
        self._train_index(feature_matrix)
        self.add_to_index(feature_matrix)


class FaissIVFFlatTrainableFeatureIndexBuilder:
    def __init__(self, batch_size: int, distance: int) -> None:
        self._batch_size = batch_size
        self._distance = distance

    def _build_index(self, num_vectors: int, vector_dim: int) -> IndexIVF:
        n_ivf = min(int(16 * np.sqrt(num_vectors)), num_vectors // 39)
        factory_string = f"IVF{n_ivf},Flat"
        index = faiss.index_factory(vector_dim, factory_string, self._distance)
        logger.debug('faiss index built by string "%s" and dimension %s', factory_string, vector_dim)
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        return index

    def build(self, num_vectors: int, vector_dim: int) -> FaissIVFTrainableFeatureIndex:
        return FaissIVFTrainableFeatureIndex(
            index=self._build_index(num_vectors, vector_dim),
            batch_size=self._batch_size,
        )


def load_retrieve_index(filepath: Path, ratio: float, n_nearest_vectors: int) -> FaissRetrievableFeatureIndex:
    return FaissRVCRetrievableFeatureIndex(
        index=faiss.read_index(str(filepath)), ratio=ratio, n_nearest_vectors=n_nearest_vectors
    )
