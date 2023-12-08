import abc
import logging
from typing import cast, Callable

from sklearn.cluster import MiniBatchKMeans

from feature_retrieval.index import NumpyArray


logger = logging.getLogger(__name__)


class IFeatureMatrixTransform:
    """Interface for transform encoded voice feature from (n_features,vector_dim) to (m_features,vector_dim)"""

    @abc.abstractmethod
    def transform(self, matrix: NumpyArray) -> NumpyArray:
        """transform given feature matrix from (n_features,vector_dim) to (m_features,vector_dim)"""
        raise NotImplementedError


class DummyFeatureTransform(IFeatureMatrixTransform):
    """do nothing"""

    def transform(self, matrix: NumpyArray) -> NumpyArray:
        return matrix


class MinibatchKmeansFeatureTransform(IFeatureMatrixTransform):
    """replaces number of examples with k-means centroids using minibatch algorythm"""

    def __init__(self, n_clusters: int, n_parallel: int) -> None:
        self._n_clusters = n_clusters
        self._n_parallel = n_parallel

    @property
    def _batch_size(self) -> int:
        return self._n_parallel * 256

    def transform(self, matrix: NumpyArray) -> NumpyArray:
        """transform given feature matrix from (n_features,vector_dim) to (n_clusters,vector_dim)"""
        cluster = MiniBatchKMeans(
            n_clusters=self._n_clusters,
            verbose=True,
            batch_size=self._batch_size,
            compute_labels=False,
            init="k-means++",
        )
        return cast(NumpyArray, cluster.fit(matrix).cluster_centers_)


class OnConditionFeatureTransform(IFeatureMatrixTransform):
    """call given transform if condition is True else call otherwise transform"""

    def __init__(
        self,
        condition: Callable[[NumpyArray], bool],
        on_condition: IFeatureMatrixTransform,
        otherwise: IFeatureMatrixTransform,
    ) -> None:
        self._condition = condition
        self._on_condition = on_condition
        self._otherwise = otherwise

    def transform(self, matrix: NumpyArray) -> NumpyArray:
        if self._condition(matrix):
            transform_name = self._on_condition.__class__.__name__
            logger.info(f"pass condition. Transform by rule {transform_name}")
            return self._on_condition.transform(matrix)
        transform_name = self._otherwise.__class__.__name__
        logger.info(f"condition is not passed. Transform by rule {transform_name}")
        return self._otherwise.transform(matrix)
