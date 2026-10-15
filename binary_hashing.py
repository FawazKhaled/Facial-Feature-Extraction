import argparse
from pathlib import Path
import sqlite3

from lib.caching import dataset_signature, load_embeddings, store_binary_codes
from lib.dataset_reader import DEFAULT_CASIA_DIRECTORY
from lib.hashing import create_matrix, encode_embedding, ham_dis, rank_hamming


DEFAULT_BITS = 256
DEFAULT_SEED = 42


def storing_binary(cache_path, signature, records, bit_length=DEFAULT_BITS, seed=DEFAULT_SEED):
    projection = create_matrix(512, bit_length, seed)
    codes = [
        (record_key, label, encode_embedding(embedding, projection))
        for record_key, label, embedding in records
    ]
    store_binary_codes(cache_path, signature, codes, bit_length, seed)
    return codes


def main():
    parser = argparse.ArgumentParser(description="Build and search a reproducible random-projection hash baseline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CASIA_DIRECTORY, help="CASIA-WebFace folder used to identify the ArcFace cache")
    parser.add_argument("--cache", type=Path, help="Embedding and hash cache (default: data/arcface_embeddings.sqlite3)")
    parser.add_argument("--query-key", type=int, default=1, help="Cached record key to use as the query (default: 1)")
    parser.add_argument("--bits", type=int, default=DEFAULT_BITS, help="Binary code length in bits (default: 256)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic projection seed (default: 42)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of closest Hamming matches to display (default: 10)")
    parser.add_argument("--bits-cache", type=Path, help="Optional cache for storing binary codes (default: same as --cache)")
    args = parser.parse_args()

    dataset_directory = args.dataset.expanduser().resolve()
    cache_path = args.cache or dataset_directory.parent / "arcface_embeddings.sqlite3"
    bits_cache_path = args.bits_cache or cache_path
    try:
        signature = dataset_signature(dataset_directory)
        records = load_embeddings(cache_path, signature)
        codes = storing_binary(bits_cache_path, signature, records, bit_length=args.bits, seed=args.seed)
        matches, precision_at_k, average_precision = rank_hamming(codes, query_key=args.query_key, top_k=args.top_k)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))

    print(f"Stored {len(codes)} codes of {args.bits} bits ({args.bits // 8} bytes each) in {bits_cache_path}")
    print(f"Random-projection baseline for query {args.query_key}: precision@{min(args.top_k, len(codes) - 1)}= {precision_at_k:.4f}, AP={average_precision:.4f}")
    print("Rank  Record key  Identity  Hamming distance")
    for rank, (record_key, identity_label, distance) in enumerate(matches, start=1):
        print(f"{rank:>4}  {record_key:>10}  {identity_label:>8g}  {distance:>16}")


if __name__ == "__main__":
    main()
