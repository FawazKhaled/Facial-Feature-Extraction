from pathlib import Path
import sqlite3

import numpy as np

from ArcFace import EMBEDDING_DIMENSION, MODEL_NAME


def dataset_signature(dataset_directory):
    index_stat = (dataset_directory / "train.idx").stat()
    record_stat = (dataset_directory / "train.rec").stat()
    return (
        f"{MODEL_NAME}|{dataset_directory.resolve()}|"
        f"{index_stat.st_size}:{index_stat.st_mtime_ns}|"
        f"{record_stat.st_size}:{record_stat.st_mtime_ns}"
    )


def initialize_embedding_cache(cache_path):
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


def get_cached_embedding(connection, signature, record_key):
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


def cache_embeddings(connection, signature, records):
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


def load_embeddings(cache_path, signature):
    if cache_path is None or not Path(cache_path).is_file():
        raise FileNotFoundError(f"cache not found: {cache_path}")
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

    results = []
    for record_key, identity_label, embedding_blob in rows:
        embedding = np.frombuffer(embedding_blob, dtype=np.float32).copy()
        results.append((int(record_key), float(identity_label), embedding))
    return results


def initialize_hash_table(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS binary_codes (
            dataset_signature TEXT,
            projection_seed INTEGER,
            bit_length INTEGER,
            record_key INTEGER,
            identity_label REAL,
            code BLOB,
            PRIMARY KEY (dataset_signature, projection_seed, bit_length, record_key)
        )
        """
    )
    connection.commit()


def store_binary_codes(cache_path, signature, codes, bit_length, seed):
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


def store_learned_codes(cache_path, signature, codes, bit_length):
    connection = sqlite3.connect(cache_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS learned_codes (
                dataset_signature TEXT NOT NULL,
                bit_length INTEGER NOT NULL,
                record_key INTEGER NOT NULL,
                identity_label REAL NOT NULL,
                code BLOB NOT NULL,
                PRIMARY KEY (dataset_signature, bit_length, record_key)
            )
            """
        )
        connection.executemany(
            """
            INSERT OR REPLACE INTO learned_codes
                (dataset_signature, bit_length, record_key, identity_label, code)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(signature, bit_length, record_key, label, code) for record_key, label, code in codes],
        )
        connection.commit()
    finally:
        connection.close()