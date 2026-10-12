from pathlib import Path
import argparse
import sqlite3

import numpy as np

from face_comparison import DEFAULT_CASIA_DIRECTORY, dataset_signature


DEFAULT_BITS = 256
DEFAULT_SEED = 42


def create_projection_matrix(embedding_dimension: int, bit_length: int, seed: int) -> np.ndarray:
    # Create a deterministic random projection for the binary baseline.
    if embedding_dimension < 1:
        raise ValueError("embedding_dimension must be positive.")
    if bit_length < 8 or bit_length > embedding_dimension or bit_length % 8:
        raise ValueError(
            "bit_length must be a multiple of 8 between 8 and embedding_dimension."
        )

    generator = np.random.default_rng(seed)
    return generator.standard_normal(
        size=(embedding_dimension, bit_length),
        dtype=np.float32,
    )


def encode_embedding(embedding: np.ndarray, projection_matrix: np.ndarray) -> bytes:
    # Project a float embedding and pack the signs into bytes.
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size != projection_matrix.shape[0]:
        raise ValueError(
            f"Expected {projection_matrix.shape[0]} embedding values, got {vector.size}."
        )
    if not np.isfinite(vector).all():
        raise ValueError("Embedding values must all be finite.")
    if not np.isfinite(projection_matrix).all():
        raise ValueError("Projection matrix values must all be finite.")

    projected_values = vector @ projection_matrix
    bits = np.greater_equal(projected_values, 0)
    return np.packbits(bits, bitorder="little").tobytes()


def hamming_distance(first_code: bytes, second_code: bytes) -> int:
    # Count different bits between equal-length packed hash codes.
    if len(first_code) != len(second_code):
        raise ValueError("Hash codes must have the same byte length.")
    first = np.frombuffer(first_code, dtype=np.uint8)
    second = np.frombuffer(second_code, dtype=np.uint8)
    xor_values = np.bitwise_xor(first, second)
    return int(np.unpackbits(xor_values).sum())


def initialize_hash_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS binary_codes (
            dataset_signature TEXT NOT NULL,
            projection_seed INTEGER NOT NULL,
            bit_length INTEGER NOT NULL,
            record_key INTEGER NOT NULL,
            identity_label REAL NOT NULL,
            code BLOB NOT NULL,
            PRIMARY KEY (
                dataset_signature,
                projection_seed,
                bit_length,
                record_key
            )
        )
        """
    )
    connection.commit()


def load_cached_embeddings(cache_path: Path, signature: str) -> list[tuple[int, float, np.ndarray]]:
    # Load only valid embeddings associated with the selected dataset and model.
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"ArcFace embedding cache not found: {cache_path}. "
            "Run face_comparison.py on the dataset first."
        )

    with sqlite3.connect(cache_path) as connection:
        rows = connection.execute(
            """
            SELECT record_key, identity_label, embedding
            FROM embeddings
            WHERE dataset_signature = ?
            ORDER BY record_key
            """,
            (signature,),
        ).fetchall()

    if not rows:
        raise ValueError(
            "No cached embeddings match the current CASIA-WebFace dataset and "
            "ArcFace model. Run face_comparison.py first."
        )

    results: list[tuple[int, float, np.ndarray]] = []
    for record_key, identity_label, embedding_blob in rows:
        embedding = np.frombuffer(embedding_blob, dtype=np.float32).copy()
        if embedding.size != 512 or not np.isfinite(embedding).all():
            raise ValueError(
                f"Cached embedding for record {record_key} is not a valid 512-D vector."
            )
        results.append((int(record_key), float(identity_label), embedding))
    return results


def store_binary_codes(cache_path: Path, signature: str, records: list[tuple[int, float, np.ndarray]], bit_length: int = DEFAULT_BITS, seed: int = DEFAULT_SEED) -> list[tuple[int, float, bytes]]:
    # Generate and persist packed codes in the existing SQLite cache.
    projection = create_projection_matrix(512, bit_length, seed)
    codes = [
        (record_key, label, encode_embedding(embedding, projection))
        for record_key, label, embedding in records
    ]

    with sqlite3.connect(cache_path) as connection:
        initialize_hash_table(connection)
        connection.executemany(
            """
            INSERT OR REPLACE INTO binary_codes (
                dataset_signature,
                projection_seed,
                bit_length,
                record_key,
                identity_label,
                code
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (signature, seed, bit_length, record_key, label, code)
                for record_key, label, code in codes
            ],
        )
        connection.commit()
    return codes


def rank_hamming_matches(codes: list[tuple[int, float, bytes]], query_key: int, top_k: int) -> tuple[list[tuple[int, float, int]], float, float]:
    # Rank cached codes by Hamming distance and measure query precision and AP.
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    code_by_key = {record_key: (label, code) for record_key, label, code in codes}
    if query_key not in code_by_key:
        raise ValueError(f"Query record {query_key} has no stored binary code.")

    query_label, query_code = code_by_key[query_key]
    ranked = sorted(
        (
            (record_key, label, hamming_distance(query_code, code))
            for record_key, label, code in codes
            if record_key != query_key
        ),
        key=lambda match: (match[2], match[0]),
    )
    if not ranked:
        raise ValueError("At least one non-query code is required for retrieval.")

    result_count = min(top_k, len(ranked))
    relevant_total = sum(label == query_label for _, label, _ in ranked)
    relevant_seen = 0
    precision_sum = 0.0
    for rank, (_, label, _) in enumerate(ranked, start=1):
        if label == query_label:
            relevant_seen += 1
            precision_sum += relevant_seen / rank

    precision_at_k = (
        sum(label == query_label for _, label, _ in ranked[:result_count])
        / result_count
    )
    average_precision = precision_sum / relevant_total if relevant_total else 0.0
    return ranked[:top_k], precision_at_k, average_precision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and search a reproducible random-projection hash baseline."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_CASIA_DIRECTORY,
        help="CASIA-WebFace folder used to identify the ArcFace cache",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="Embedding and hash cache (default: data/arcface_embeddings.sqlite3)",
    )
    parser.add_argument(
        "--query-key",
        type=int,
        default=1,
        help="Cached record key to use as the query (default: 1)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=DEFAULT_BITS,
        help="Binary code length in bits (default: 256)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic projection seed (default: 42)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of closest Hamming matches to display (default: 10)",
    )
    args = parser.parse_args()

    dataset_directory = args.dataset.expanduser().resolve()
    cache_path = args.cache or dataset_directory.parent / "arcface_embeddings.sqlite3"
    try:
        if args.top_k < 1:
            raise ValueError("--top-k must be at least 1.")
        signature = dataset_signature(dataset_directory)
        records = load_cached_embeddings(cache_path, signature)
        codes = store_binary_codes(
            cache_path,
            signature,
            records,
            bit_length=args.bits,
            seed=args.seed,
        )
        matches, precision_at_k, average_precision = rank_hamming_matches(
            codes,
            query_key=args.query_key,
            top_k=args.top_k,
        )
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))

    print(
        f"Stored {len(codes)} codes of {args.bits} bits "
        f"({args.bits // 8} bytes each) in {cache_path}"
    )
    print(
        f"Random-projection baseline for query {args.query_key}: "
        f"precision@{min(args.top_k, len(codes) - 1)}="
        f"{precision_at_k:.4f}, AP={average_precision:.4f}"
    )
    print("Rank  Record key  Identity  Hamming distance")
    for rank, (record_key, identity_label, distance) in enumerate(matches, start=1):
        print(f"{rank:>4}  {record_key:>10}  {identity_label:>8g}  {distance:>16}")


if __name__ == "__main__":
    main()
