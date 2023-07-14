import argparse
import logging
from pathlib import Path

import numpy as np
import faiss

from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_vectors_matrix(vectors_dir: Path) -> np.array:
    logger.info("start get vectors from %s", vectors_dir.absolute())
    vectors_matrices = []
    for vectors in vectors_dir.rglob("*.npy"):
        vectors_matrices.append(np.load(str(vectors)))
    logger.info("done. fetched %s files", len(vectors_matrices))
    matrix = np.concatenate(vectors_matrices, axis=0)
    logger.debug("final matrix shape: %s", matrix.shape)
    return matrix


def create_faiss_index(num_vectors: int, vector_dim: int):
    n_ivf = min(int(16 * np.sqrt(num_vectors)), num_vectors // 39)
    factory_string = f"IVF{n_ivf},Flat"
    logger.debug("start creating faiss index of dimension %s by rule \"%s\"", vector_dim, factory_string)
    index = faiss.index_factory(vector_dim, factory_string)
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    return index


def create_vector_index(matrix: np.array):
    num_vectors, vector_dim = matrix.shape
    index = create_faiss_index(num_vectors, vector_dim)
    logger.debug("start train index")
    index.train(matrix)
    return index


def batch_add_to_index(matrix: np.array, index, batch_size: int):
    logger.info("start adding vectors to index by batch size %s", batch_size)
    vectors_count = matrix.shape[0]
    n_batches = vectors_count // batch_size
    for batch in tqdm(np.array_split(matrix, n_batches, axis=0), total=n_batches):
        index.add(batch)


def train_index(vectors_dir: Path):
    batch_size = 8192
    vectors_matrix = create_vectors_matrix(vectors_dir)
    index = create_vector_index(vectors_matrix)
    batch_add_to_index(vectors_matrix, index, batch_size)
    return index


def train_whisper_index(base_path: Path, speaker: str):
    logger.info("start train whisper index for speaker \"%s\"", speaker)
    whisper_vectors_path = base_path / "data_svc" / "whisper" / speaker
    return train_index(whisper_vectors_path)


def save_whisper_index(index, speaker_indexes_dir: Path) -> None:
    index_filepath = speaker_indexes_dir / "whisper.index"
    logger.debug("save whisper index to \"%s\"", index_filepath)
    faiss.write_index(index, str(index_filepath))


def train_hubert_index(base_path: Path, speaker: str):
    logger.info("start train hubert index for speaker \"%s\"", speaker)
    hubert_vectors_path = base_path / "data_svc" / "hubert" / speaker
    return train_index(hubert_vectors_path)


def save_hubert_index(index, speaker_indexes_dir: Path) -> None:
    index_filepath = speaker_indexes_dir / "hubert.index"
    logger.debug("save hubert index to \"%s\"", index_filepath)
    faiss.write_index(index, str(index_filepath))


def create_index_for_speaker(base_path: Path, indexes_path: Path, speaker: str) -> None:
    logger.info("start create faiss index for speaker \"%s\"", speaker)
    speaker_dir_path = indexes_path / speaker
    logger.debug("create index speaker folder \"%s\"", speaker_dir_path)
    speaker_dir_path.mkdir()
    logger.debug("start create whisper index")
    whisper_index = train_whisper_index(base_path, speaker)
    save_whisper_index(whisper_index, speaker_dir_path)
    del whisper_index
    logger.debug("done")
    logger.debug("start create hubert index")
    hubert_index = train_hubert_index(base_path, speaker)
    save_hubert_index(hubert_index, speaker_dir_path)
    del hubert_index
    logger.debug("done")
    logger.debug("faiss index created for speaker \"%s\"", speaker)


def get_speaker_list(base_path: Path):
    speakers_path = base_path / "data_svc" / "waves-16k"
    return [speaker_dir.name for speaker_dir in speakers_path.iterdir() if speaker_dir.is_dir()]


def create_indexes_path(base_path: Path) -> Path:
    indexes_path = base_path / "data_svc" / "indexes"
    logger.info("create indexes folder %s", indexes_path)
    indexes_path.mkdir(exist_ok=True)
    return indexes_path


def main() -> None:
    arg_parser = argparse.ArgumentParser("crate faiss indexes for feature retrieval")
    arg_parser.add_argument("--debug", action="store_true")
    args = arg_parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    base_path = Path(".").absolute()
    speakers = get_speaker_list(base_path)
    logger.info("got %s speakers: %s", len(speakers), speakers)
    indexes_path = create_indexes_path(base_path)
    for speaker in speakers:
        create_index_for_speaker(base_path, indexes_path, speaker)
    logger.info("done!")


if __name__ == '__main__':
    main()
