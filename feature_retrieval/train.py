from pathlib import Path
from typing import cast

import numpy as np

from feature_retrieval import NumpyArray
from feature_retrieval.index import FaissIVFFlatTrainableFeatureIndexBuilder, logger
from feature_retrieval.transform import IFeatureMatrixTransform


def train_index(
    features_path: Path,
    index_save_filepath: Path,
    index_builder: FaissIVFFlatTrainableFeatureIndexBuilder,
    feature_transform: IFeatureMatrixTransform,
) -> None:
    logger.info("start getting feature vectors from %s", features_path.absolute())
    feature_matrix = get_feature_matrix(features_path)
    logger.debug("fetched %s features", feature_matrix.shape[0])

    logger.info("apply transform to feature matrix")
    feature_matrix = feature_transform.transform(feature_matrix)
    num_vectors, vector_dim = feature_matrix.shape
    logger.debug("features transformed. Current features %s", num_vectors)

    feature_index = index_builder.build(num_vectors=num_vectors, vector_dim=vector_dim)
    logger.info("adding features to index with training")

    feature_index.add_with_train(feature_matrix)
    feature_index.save(index_save_filepath)
    logger.info("index saved to %s", index_save_filepath.absolute())


def get_feature_matrix(features_dir_path: Path) -> NumpyArray:
    matrices = [np.load(str(features_path)) for features_path in features_dir_path.rglob("*.npy")]
    feature_matrix = np.concatenate(matrices, axis=0)
    return cast(NumpyArray, feature_matrix)
