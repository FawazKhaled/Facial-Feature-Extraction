from collections.abc import Sequence
from importlib import import_module
import argparse
import heapq
import logging
from pathlib import Path
import sqlite3
from typing import Protocol, cast

import numpy as np
from PIL import Image

from image_input import (
    DEFAULT_CASIA_DIRECTORY,
    iter_casia_index,
    load_casia_record,
)


EMBEDDING_DIMENSION = 512
MODEL_NAME = "buffalo_l"
DEFAULT_TOP_K = 10
CACHE_BATCH_SIZE = 512
logger = logging.getLogger(__name__)


class DetectedFace(Protocol):
    embedding: np.ndarray


class ArcFaceModel(Protocol):
    def prepare(self, ctx_id: int = ..., det_size: tuple[int, int] = ...) -> None: ...

    def get(self, img: np.ndarray) -> Sequence[DetectedFace]: ...


def load_arcface_model() -> ArcFaceModel:
    # Load InsightFace's pretrained ArcFace model using CPU inference.
    try:
        insightface_app = import_module("insightface.app")
    except ImportError as error:
        raise RuntimeError(
            "ArcFace extraction requires InsightFace and ONNX Runtime. "
            "Install them with: pip install insightface onnxruntime"
        ) from error

    face_analysis = getattr(insightface_app, "FaceAnalysis")
    model = face_analysis(
        name=MODEL_NAME,
        providers=["CPUExecutionProvider"],
    )
    model.prepare(ctx_id=-1)
    return cast(ArcFaceModel, model)


def extract_arcface_embedding(model: ArcFaceModel, image: Image.Image) -> np.ndarray:
    # Detect one face and return its normalized 512-D ArcFace feature vector.
    rgb_image = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr_image = rgb_image[:, :, ::-1].copy()
    faces = model.get(bgr_image)

    if len(faces) != 1:
        raise ValueError(
            f"Expected exactly one face in the image, but detected {len(faces)}."
        )

    embedding = np.asarray(faces[0].embedding, dtype=np.float32).reshape(-1)
    if embedding.size != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected a {EMBEDDING_DIMENSION}-D ArcFace embedding, "
            f"received {embedding.size} values."
        )

    norm = np.linalg.norm(embedding)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("ArcFace returned an invalid or zero-length embedding.")

    return embedding / norm


def cosine_similarity(first_embedding: np.ndarray, second_embedding: np.ndarray) -> float:
    # Return cosine similarity for two finite, non-zero 512-D embeddings.
    first = np.asarray(first_embedding, dtype=np.float32).reshape(-1)
    second = np.asarray(second_embedding, dtype=np.float32).reshape(-1)
    if first.size != EMBEDDING_DIMENSION or second.size != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Cosine similarity requires two {EMBEDDING_DIMENSION}-D embeddings."
        )
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Embeddings must contain only finite values.")

    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm == 0 or second_norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero-length embedding.")

    similarity = float(np.dot(first, second) / (first_norm * second_norm))
    return float(np.clip(similarity, -1.0, 1.0))


def find_record_offset(dataset_directory: Path, record_key: int) -> int:
    # Find the byte offset of a record key in train.idx.
    for key, offset in iter_casia_index(dataset_directory):
        if key == record_key:
            return offset
    raise ValueError(f"CASIA-WebFace record key {record_key} is not in train.idx.")


def dataset_signature(dataset_directory: Path) -> str:
    # Identify the dataset/model pair so stale embeddings are not reused.
    index_stat = (dataset_directory / "train.idx").stat()
    record_stat = (dataset_directory / "train.rec").stat()
    return (
        f"{MODEL_NAME}|{dataset_directory.resolve()}|"
        f"{index_stat.st_size}:{index_stat.st_mtime_ns}|"
        f"{record_stat.st_size}:{record_stat.st_mtime_ns}"
    )


def initialize_embedding_cache(cache_path: Path) -> sqlite3.Connection:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            dataset_signature TEXT NOT NULL,
            record_key INTEGER NOT NULL,
            identity_label REAL NOT NULL,
            embedding BLOB NOT NULL,
            PRIMARY KEY (dataset_signature, record_key)
        )
        """
    )
    connection.commit()
    return connection


def get_cached_embedding(
    connection: sqlite3.Connection,
    signature: str,
    record_key: int,
) -> tuple[float, np.ndarray] | None:
    row = connection.execute(
        """
        SELECT identity_label, embedding
        FROM embeddings
        WHERE dataset_signature = ? AND record_key = ?
        """,
        (signature, record_key),
    ).fetchone()
    if row is None:
        return None

    identity_label = float(row[0])
    embedding = np.frombuffer(row[1], dtype=np.float32).copy()
    if embedding.size != EMBEDDING_DIMENSION or not np.isfinite(embedding).all():
        raise ValueError(f"Invalid cached embedding for CASIA record {record_key}.")
    return identity_label, embedding


def cache_embeddings(connection: sqlite3.Connection, signature: str, records: list[tuple[int, float, np.ndarray]]) -> None:
    connection.executemany(
        """
        INSERT OR REPLACE INTO embeddings
            (dataset_signature, record_key, identity_label, embedding)
        VALUES (?, ?, ?, ?)
        """,
        [
            (signature, key, label, embedding.astype(np.float32, copy=False).tobytes())
            for key, label, embedding in records
        ],
    )
    connection.commit()
    records.clear()


def rank_casia_records(model: ArcFaceModel, dataset_directory: Path, query_key: int, query_embedding: np.ndarray, connection: sqlite3.Connection, signature: str, top_k: int, limit: int | None = None) -> list[tuple[int, float, float]]:
    # Stream CASIA records and retain only the highest-scoring top-k matches.
    matches: list[tuple[float, int, float]] = []
    pending_embeddings: list[tuple[int, float, np.ndarray]] = []
    candidate_count = 0

    for record_key, record_offset in iter_casia_index(dataset_directory):
        if record_key == query_key:
            continue
        if limit is not None and candidate_count >= limit:
            break
        candidate_count += 1

        cached = get_cached_embedding(connection, signature, record_key)
        if cached is None:
            try:
                image, identity_label = load_casia_record(
                    dataset_directory,
                    record_key,
                    record_offset,
                )
                embedding = extract_arcface_embedding(model, image)
            except (OSError, ValueError) as error:
                logger.warning("Skipping CASIA record %s: %s", record_key, error)
                continue
            pending_embeddings.append((record_key, identity_label, embedding))
        else:
            identity_label, embedding = cached

        similarity = cosine_similarity(query_embedding, embedding)
        candidate = (similarity, -record_key, identity_label)
        if len(matches) < top_k:
            heapq.heappush(matches, candidate)
        elif candidate > matches[0]:
            heapq.heapreplace(matches, candidate)

        if len(pending_embeddings) >= CACHE_BATCH_SIZE:
            cache_embeddings(connection, signature, pending_embeddings)

        if candidate_count % 1000 == 0:
            logger.info("Scanned %s CASIA records", candidate_count)

    if pending_embeddings:
        cache_embeddings(connection, signature, pending_embeddings)

    return [
        (-negative_key, identity_label, similarity)
        for similarity, negative_key, identity_label in sorted(matches, reverse=True)
    ]


def run_comparison(dataset_directory: Path, query_key: int, top_k: int, cache_path: Path, limit: int | None = None) -> list[tuple[int, float, float]]:
    # Load the query, create/reuse embeddings, and rank the dataset.
    dataset_directory = dataset_directory.expanduser().resolve()
    if not dataset_directory.is_dir():
        raise NotADirectoryError(f"CASIA-WebFace folder not found: {dataset_directory}")
    if not (dataset_directory / "train.rec").is_file():
        raise FileNotFoundError(f"CASIA-WebFace records not found in {dataset_directory}")
    if not (dataset_directory / "train.idx").is_file():
        raise FileNotFoundError(f"CASIA-WebFace index not found in {dataset_directory}")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1 when specified.")

    query_offset = find_record_offset(dataset_directory, query_key)
    signature = dataset_signature(dataset_directory)
    connection = initialize_embedding_cache(cache_path)
    try:
        model = load_arcface_model()
        cached_query = get_cached_embedding(connection, signature, query_key)
        if cached_query is None:
            query_image, query_label = load_casia_record(
                dataset_directory,
                query_key,
                query_offset,
            )
            query_embedding = extract_arcface_embedding(model, query_image)
            cache_embeddings(
                connection,
                signature,
                [(query_key, query_label, query_embedding)],
            )
        else:
            query_label, query_embedding = cached_query

        logger.info(
            "Query record %s has identity label %g",
            query_key,
            query_label,
        )
        return rank_casia_records(
            model,
            dataset_directory,
            query_key,
            query_embedding,
            connection,
            signature,
            top_k,
            limit,
        )
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search CASIA-WebFace for faces similar to a selected record."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_CASIA_DIRECTORY,
        help="CASIA-WebFace folder containing train.idx and train.rec",
    )
    parser.add_argument(
        "--query-key",
        type=int,
        default=1,
        help="Record key to use as the query face (default: 1)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of highest-similarity matches to display (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of other records to scan (useful for a quick test)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="Embedding cache database (default: data/arcface_embeddings.sqlite3)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset_directory = args.dataset.expanduser().resolve()
    cache_path = args.cache or dataset_directory.parent / "arcface_embeddings.sqlite3"
    try:
        matches = run_comparison(
            dataset_directory,
            args.query_key,
            args.top_k,
            cache_path,
            args.limit,
        )
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))

    if not matches:
        parser.error("No dataset records could be compared successfully.")

    print("Rank  Record key  Identity  Cosine similarity")
    for rank, (record_key, identity_label, similarity) in enumerate(matches, start=1):
        print(
            f"{rank:>4}  {record_key:>10}  {identity_label:>8g}  "
            f"{similarity:>17.4f}"
        )


if __name__ == "__main__":
    main()
