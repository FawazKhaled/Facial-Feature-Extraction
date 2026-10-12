import argparse
import logging
from pathlib import Path
import sqlite3

from ArcFace import extract_arcface_embedding, load_arcface_model
from caching import (
    cache_embeddings,
    dataset_signature,
    get_cached_embedding,
    initialize_embedding_cache,
)
from dataset_reader import DEFAULT_CASIA_DIRECTORY, load_casia_record
from retrieval import find_record_offset, rank_casia_records


DEFAULT_TOP_K = 10


def run_comparison(dataset_directory, query_key, top_k, cache_path, limit=None):
    dataset_directory = dataset_directory.expanduser().resolve()
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
            query_image, query_label = load_casia_record(dataset_directory, query_key, query_offset)
            query_embedding = extract_arcface_embedding(model, query_image)
            cache_embeddings(connection, signature, [(query_key, query_label, query_embedding)])
        else:
            query_label, query_embedding = cached_query

        logging.info("Query record %s has identity label %g", query_key, query_label)
        return rank_casia_records(model, dataset_directory, query_key, query_embedding, connection, signature, top_k, limit)
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description="Search CASIA-WebFace for faces similar to a selected record.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CASIA_DIRECTORY, help="CASIA-WebFace folder containing train.idx and train.rec")
    parser.add_argument("--query-key", type=int, default=1, help="Record key to use as the query face (default: 1)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help=f"Number of highest-similarity matches to display (default: {DEFAULT_TOP_K})")
    parser.add_argument("--limit", type=int, help="Optional maximum number of other records to scan (useful for a quick test)")
    parser.add_argument("--cache", type=Path, help="Embedding cache database (default: data/arcface_embeddings.sqlite3)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset_directory = args.dataset.expanduser().resolve()
    cache_path = args.cache or dataset_directory.parent / "arcface_embeddings.sqlite3"
    try:
        matches = run_comparison(dataset_directory, args.query_key, args.top_k, cache_path, args.limit)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))

    if not matches:
        parser.error("No dataset records could be compared successfully.")

    print("Rank  Record key  Identity  Cosine similarity")
    for rank, (record_key, identity_label, similarity) in enumerate(matches, start=1):
        print(f"{rank:>4}  {record_key:>10}  {identity_label:>8g}  {similarity:>17.4f}")


if __name__ == "__main__":
    main()
