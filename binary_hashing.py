from pathlib import Path
import argparse
import sqlite3
import numpy as np
from face_comparison import DEFAULT_CASIA_DIRECTORY, dataset_signature


DEFAULT_BITS = 256
DEFAULT_SEED = 42


def create_matrix(embedding_dimension, bit_length, seed):

    generator = np.random.default_rng(seed)
    return generator.standard_normal(size=(embedding_dimension, bit_length),dtype=np.float32)


def encode_embedding(embedding, projection_matrix):
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)

    projected_values = vector @ projection_matrix
    bits = np.greater_equal(projected_values, 0)
    return np.packbits(bits, bitorder="little").tobytes()


def ham_dis(first_code, second_code):
    first = np.frombuffer(first_code, dtype=np.uint8)
    second = np.frombuffer(second_code, dtype=np.uint8)
    xor_values = np.bitwise_xor(first, second)
    return int(np.unpackbits(xor_values).sum())


def initialize_hash_table(connection):
    connection.execute("""
        CREATE TABLE IF NOT EXISTS binary_codes (
            dataset_signature TEXT,
            projection_seed INTEGER,
            bit_length INTEGER,
            record_key INTEGER,
            identity_label REAL,
            code BLOB,
            PRIMARY KEY (dataset_signature, projection_seed, bit_length, record_key)
        )
    """)
    connection.commit()

def load_embeddings(cache_path, signature):
    if cache_path is None or not Path(cache_path).is_file():
        raise FileNotFoundError(f"cache not found: {cache_path}")
    with sqlite3.connect(cache_path) as connection:
        rows = connection.execute("""SELECT record_key, identity_label, embedding FROM embeddings WHERE dataset_signature = ? ORDER BY record_key""",(signature,)).fetchall()

    results = []
    for record_key, identity_label, embedding_blob in rows:
        embedding = np.frombuffer(embedding_blob, dtype=np.float32).copy()
        results.append((int(record_key), float(identity_label), embedding))
    return results


def storing_binary(cache_path, signature, records, bit_length = DEFAULT_BITS, seed = DEFAULT_SEED):
    projection = create_matrix(512, bit_length, seed)
    codes = [(record_key, label, encode_embedding(embedding, projection)) for record_key, label, embedding in records]

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


def rank_hamming(codes, query_key, top_k):
    code_by_key = {record_key: (label, code) for record_key, label, code in codes}

    if query_key not in code_by_key:
        raise ValueError(f"Query key {query_key} not found in the provided codes.")
    else:
        query_label, query_code = code_by_key[query_key]
    ranked = sorted(((record_key, label, ham_dis(query_code, code)) for record_key, label, code in codes if record_key != query_key), key=lambda match: (match[2], match[0]))

    result_count = min(top_k, len(ranked))
    relevant_total = sum(label == query_label for _, label, _ in ranked)
    relevant_seen = 0
    precision_sum = 0.0
    for rank, (_, label, _) in enumerate(ranked, start=1):
        if label == query_label:
            relevant_seen += 1
            precision_sum += relevant_seen / rank

    precision_at_k = (sum(label == query_label for _, label, _ in ranked[:result_count]) / result_count) if result_count else 0.0
    if relevant_total == 0:
        average_precision = 0.0
    else:
        average_precision = precision_sum / relevant_total
    return ranked[:top_k], precision_at_k, average_precision


def main():
    parser = argparse.ArgumentParser(description="Build and search a reproducible random-projection hash baseline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CASIA_DIRECTORY, help="CASIA-WebFace folder used to identify the ArcFace cache")
    parser.add_argument("--cache", type=Path, help="Embedding and hash cache (default: data/arcface_embeddings.sqlite3)")
    parser.add_argument("--query-key", type=int, default=1, help="Cached record key to use as the query (default: 1)")
    parser.add_argument("--bits", type=int, default=DEFAULT_BITS, help="Binary code length in bits (default: 256)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic projection seed (default: 42)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of closest Hamming matches to display (default: 10)")
    args = parser.parse_args()

    dataset_directory = args.dataset.expanduser().resolve()
    cache_path = args.cache or dataset_directory.parent / "arcface_embeddings.sqlite3"
    try:
        signature = dataset_signature(dataset_directory)
        records = load_embeddings(cache_path, signature)
        codes = storing_binary(cache_path, signature, records, bit_length=args.bits, seed=args.seed)
        matches, precision_at_k, average_precision = rank_hamming(codes, query_key=args.query_key, top_k=args.top_k)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))

    print(f"Stored {len(codes)} codes of {args.bits} bits ({args.bits // 8} bytes each) in {cache_path}")
    print(f"Random-projection baseline for query {args.query_key}: precision@{min(args.top_k, len(codes) - 1)}= {precision_at_k:.4f}, AP={average_precision:.4f}")
    print("Rank  Record key  Identity  Hamming distance")
    for rank, (record_key, identity_label, distance) in enumerate(matches, start=1):
        print(f"{rank:>4}  {record_key:>10}  {identity_label:>8g}  {distance:>16}")


if __name__ == "__main__":
    main()