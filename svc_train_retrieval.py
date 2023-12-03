import argparse
import logging
import multiprocessing
from functools import partial
from pathlib import Path

import faiss

from feature_retrieval import (
    train_index,
    FaissIVFFlatTrainableFeatureIndexBuilder,
    OnConditionFeatureTransform,
    MinibatchKmeansFeatureTransform,
    DummyFeatureTransform,
)

logger = logging.getLogger(__name__)


def get_speaker_list(base_path: Path):
    speakers_path = base_path / "waves-16k"
    if not speakers_path.exists():
        raise FileNotFoundError(f"path {speakers_path} does not exists")
    return [speaker_dir.name for speaker_dir in speakers_path.iterdir() if speaker_dir.is_dir()]


def create_indexes_path(base_path: Path) -> Path:
    indexes_path = base_path / "indexes"
    logger.info("create indexes folder %s", indexes_path)
    indexes_path.mkdir(exist_ok=True)
    return indexes_path


def create_index(
        feature_name: str,
        prefix: str,
        speaker: str,
        base_path: Path,
        indexes_path: Path,
        compress_features_after: int,
        n_clusters: int,
        n_parallel: int,
        train_batch_size: int = 8192,
) -> None:
    features_path = base_path / feature_name / speaker
    if not features_path.exists():
        raise ValueError(f'features not found by path {features_path}')
    index_path = indexes_path / speaker
    index_path.mkdir(exist_ok=True)
    index_filename = f"{prefix}{feature_name}.index"
    index_filepath = index_path / index_filename
    logger.debug('index will be save to %s', index_filepath)

    builder = FaissIVFFlatTrainableFeatureIndexBuilder(train_batch_size, distance=faiss.METRIC_L2)
    transform = OnConditionFeatureTransform(
        condition=lambda matrix: matrix.shape[0] > compress_features_after,
        on_condition=MinibatchKmeansFeatureTransform(n_clusters, n_parallel),
        otherwise=DummyFeatureTransform()
    )
    train_index(features_path, index_filepath, builder, transform)


def main() -> None:
    arg_parser = argparse.ArgumentParser("crate faiss indexes for feature retrieval")
    arg_parser.add_argument("--debug", action="store_true")
    arg_parser.add_argument("--prefix", default='', help="add prefix to index filename")
    arg_parser.add_argument('--speakers', nargs="+",
                            help="speaker names to create an index. By default all speakers are from data_svc")
    arg_parser.add_argument("--compress-features-after", type=int, default=200_000,
                            help="If the number of features is greater than the value compress "
                                 "feature vectors using MiniBatchKMeans.")
    arg_parser.add_argument("--n-clusters", type=int, default=10_000,
                            help="Number of centroids to which features will be compressed")

    arg_parser.add_argument("--n-parallel", type=int, default=multiprocessing.cpu_count()-1,
                            help="Nuber of parallel job of MinibatchKmeans. Default is cpus-1")
    args = arg_parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    base_path = Path(".").absolute() / "data_svc"
    if args.speakers:
        speakers = args.speakers
    else:
        speakers = get_speaker_list(base_path)

    logger.info("got %s speakers: %s", len(speakers), speakers)
    indexes_path = create_indexes_path(base_path)

    create_index_func = partial(
        create_index,
        prefix=args.prefix,
        base_path=base_path,
        indexes_path=indexes_path,
        compress_features_after=args.compress_features_after,
        n_clusters=args.n_clusters,
        n_parallel=args.n_parallel,
    )

    for speaker in speakers:
        logger.info("create hubert index for speaker %s", speaker)
        create_index_func(feature_name="hubert", speaker=speaker)

        logger.info("create whisper index for speaker %s", speaker)
        create_index_func(feature_name="whisper", speaker=speaker)

    logger.info("done!")


if __name__ == '__main__':
    main()
