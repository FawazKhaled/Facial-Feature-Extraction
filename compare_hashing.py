import argparse
from pathlib import Path
import sqlite3

import torch

from caching import dataset_signature
from dataset_reader import DEFAULT_CASIA_DIRECTORY
from hashing import create_matrix, encode_embedding
from hashing_cnn import encode_hash, evaluate_binary_codes, load_hashing_model, split_cached_embeddings


def main():
    parser = argparse.ArgumentParser(description="Compare learned and random-projection hashes on one held-out split.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CASIA_DIRECTORY, help="CASIA-WebFace folder used to identify the embedding cache")
    parser.add_argument("--cache", type=Path, help="ArcFace embedding cache (default: data/arcface_embeddings.sqlite3)")
    parser.add_argument("--model", type=Path, help="Trained hashing model (default: data/hashing_model.pt)")
    parser.add_argument("--bits", type=int, default=256, help="Number of bits used by the random-projection baseline")
    parser.add_argument("--top-k", type=int, default=10, help="Number of nearest results used for precision@k")
    parser.add_argument("--eval-ratio", type=float, default=0.2, help="Fraction of each identity held out for evaluation")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed used for the training/evaluation split")
    parser.add_argument("--projection-seed", type=int, default=42, help="Seed used for the random projection")
    parser.add_argument("--device", help="PyTorch device, such as cpu or cuda")
    args = parser.parse_args()

    dataset_directory = args.dataset.expanduser().resolve()
    cache_path = args.cache or dataset_directory.parent / "arcface_embeddings.sqlite3"
    model_path = args.model or dataset_directory.parent / "hashing_model.pt"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    try:
        if args.bits < 1 or args.top_k < 1:
            raise ValueError("bits and top-k must be at least 1.")
        signature = dataset_signature(dataset_directory)
        _, _, eval_embeddings, eval_labels = split_cached_embeddings(cache_path, signature, args.eval_ratio, args.split_seed)
        model = load_hashing_model(model_path, device)
        model_bits = model[2].out_features
        if model_bits != args.bits:
            raise ValueError(f"The trained model uses {model_bits} bits, but --bits is {args.bits}.")

        learned_codes = encode_hash(model, eval_embeddings, device)
        projection = create_matrix(eval_embeddings.shape[1], args.bits, args.projection_seed)
        random_codes = [encode_embedding(embedding, projection) for embedding in eval_embeddings]
        learned_precision, learned_ap, learned_queries = evaluate_binary_codes(learned_codes, eval_labels, args.top_k)
        random_precision, random_ap, random_queries = evaluate_binary_codes(random_codes, eval_labels, args.top_k)
        if learned_queries != random_queries:
            raise RuntimeError("The two methods were evaluated on different query counts.")
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))

    print(f"Same evaluation split: {len(eval_embeddings)} embeddings, {learned_queries} queries, {args.bits} bits.")
    print(f"Learned hash: precision@{args.top_k}={learned_precision:.4f}, AP={learned_ap:.4f}.")
    print(f"Random projection (seed {args.projection_seed}): precision@{args.top_k}={random_precision:.4f}, AP={random_ap:.4f}.")


if __name__ == "__main__":
    main()
