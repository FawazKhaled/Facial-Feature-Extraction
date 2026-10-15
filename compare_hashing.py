import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
import torch

from lib.caching import dataset_signature
from lib.dataset_reader import DEFAULT_CASIA_DIRECTORY
from lib.hashing import create_matrix, encode_embedding
from hashing_cnn import encode_hash, evaluate_binary_codes, load_hashing_model, split_cached_embeddings


# full float ArcFace search, used as the upper bound for any binary code
def evaluate_float_embeddings(embeddings, labels, top_k=10):
    values = np.asarray(embeddings, dtype=np.float32)
    values = values / np.linalg.norm(values, axis=1, keepdims=True)
    labels = np.asarray(labels)
    precisions = []
    average_precisions = []

    for query_index, query_label in enumerate(labels):
        other_indexes = np.arange(len(labels)) != query_index
        similarities = values[other_indexes] @ values[query_index]
        order = np.lexsort((np.flatnonzero(other_indexes), -similarities))
        relevant = labels[other_indexes][order] == query_label
        relevant_count = int(np.count_nonzero(labels == query_label) - 1)
        if relevant_count == 0:
            continue

        result_count = min(top_k, len(order))
        precisions.append(np.count_nonzero(relevant[:result_count]) / result_count if result_count else 0.0)
        ranks = np.arange(1, len(order) + 1)
        cumulative_precision = np.cumsum(relevant) / ranks
        average_precisions.append(float(np.sum(cumulative_precision[relevant]) / relevant_count))

    if not precisions:
        raise ValueError("Evaluation split has no identity with multiple samples.")
    return float(np.mean(precisions)), float(np.mean(average_precisions)), len(precisions)


# iterative quantization (Gong & Lazebnik): PCA down to bit_length, then learn a rotation that reduces quantization error
def train_itq(embeddings, bit_length=256, iterations=50, seed=42):
    values = np.asarray(embeddings, dtype=np.float32)
    if bit_length > values.shape[1]:
        raise ValueError(f"ITQ needs bits <= embedding size ({values.shape[1]}), but got {bit_length}.")

    mean = values.mean(axis=0)
    centered = values - mean
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    pca = components[:bit_length].T
    projected = centered @ pca
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.standard_normal((bit_length, bit_length)))

    for _ in range(iterations):
        codes = np.where(projected @ rotation >= 0, 1.0, -1.0)
        left, _, right = np.linalg.svd(projected.T @ codes)
        rotation = left @ right

    return mean, (pca @ rotation).astype(np.float32)


def encode_itq(embeddings, mean, projection):
    codes = (np.asarray(embeddings, dtype=np.float32) - mean) @ projection >= 0
    packed_codes = np.packbits(codes, axis=1, bitorder="little")
    return [code.tobytes() for code in packed_codes]


def main():
    parser = argparse.ArgumentParser(description="Compare learned and random-projection hashes on one held-out split.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CASIA_DIRECTORY, help="CASIA-WebFace folder used to identify the embedding cache")
    parser.add_argument("--cache", type=Path, help="ArcFace embedding cache (default: data/arcface_embeddings.sqlite3)")
    parser.add_argument("--model", type=Path, help="Trained hashing model (default: data/hashing_model.pt)")
    parser.add_argument("--trials-dir", type=Path, help="Directory containing trials.json and trial models")
    parser.add_argument("--top-trials", type=int, help="Evaluate this many top validation-ranked trial models instead of --model")
    parser.add_argument("--bits", type=int, default=256, help="Number of bits used by the random-projection baseline")
    parser.add_argument("--top-k", type=int, default=10, help="Number of nearest results used for precision@k")
    parser.add_argument("--validation-ratio", type=float, default=0.2, help="Fraction of each identity held out for validation")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of each identity held out for final testing")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed used for the training/validation/test split")
    parser.add_argument("--projection-seed", type=int, default=42, help="Seed used for the random projection")
    parser.add_argument("--device", help="PyTorch device, such as cpu or cuda")
    args = parser.parse_args()

    dataset_directory = args.dataset.expanduser().resolve()
    cache_path = args.cache or dataset_directory.parent / "arcface_embeddings.sqlite3"
    model_path = args.model or dataset_directory.parent / "hashing_model.pt"
    trials_dir = args.trials_dir or dataset_directory.parent / "hashing_trials"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    try:
        if args.bits < 1 or args.top_k < 1 or (args.top_trials is not None and args.top_trials < 1):
            raise ValueError("bits, top-k, and top-trials must be at least 1.")
        signature = dataset_signature(dataset_directory)
        if args.top_trials is not None:
            manifest_path = trials_dir / "trials.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["dataset_signature"] != signature:
                raise ValueError("Trial manifest belongs to a different dataset.")
            if manifest["bits"] != args.bits:
                raise ValueError(f"Trial models use {manifest['bits']} bits, but --bits is {args.bits}.")
            train_embeddings, _, _, _, test_embeddings, test_labels = split_cached_embeddings(cache_path, signature, manifest["validation_ratio"], manifest["test_ratio"], manifest["split_seed"])
            selected_trials = sorted(manifest["trials"], key=lambda trial: (trial["validation_precision_at_10"], trial["validation_average_precision"]), reverse=True)[:args.top_trials]
            if not selected_trials:
                raise ValueError("The trial manifest contains no trained trials.")
        else:
            train_embeddings, _, _, _, test_embeddings, test_labels = split_cached_embeddings(cache_path, signature, args.validation_ratio, args.test_ratio, args.split_seed)
            selected_trials = [{"model_path": str(model_path), "trial": None, "model_seed": None, "epochs": None}]

        projection = create_matrix(test_embeddings.shape[1], args.bits, args.projection_seed)
        random_codes = [encode_embedding(embedding, projection) for embedding in test_embeddings]
        random_precision, random_ap, random_queries = evaluate_binary_codes(random_codes, test_labels, args.top_k)
        float_precision, float_ap, float_queries = evaluate_float_embeddings(test_embeddings, test_labels, args.top_k)
        itq_mean, itq_projection = train_itq(train_embeddings, args.bits, seed=args.projection_seed)
        itq_precision, itq_ap, itq_queries = evaluate_binary_codes(encode_itq(test_embeddings, itq_mean, itq_projection), test_labels, args.top_k)
        if float_queries != random_queries or itq_queries != random_queries:
            raise RuntimeError("The baselines were evaluated on different query counts.")
        trial_scores = []
        for trial in selected_trials:
            model = load_hashing_model(trial["model_path"], device)
            model_bits = model[2].out_features
            if model_bits != args.bits:
                raise ValueError(f"Trial {trial.get('trial')} uses {model_bits} bits, but --bits is {args.bits}.")
            learned_codes = encode_hash(model, test_embeddings, device)
            precision, average_precision, learned_queries = evaluate_binary_codes(learned_codes, test_labels, args.top_k)
            if learned_queries != random_queries:
                raise RuntimeError("The learned and random-projection methods were evaluated on different query counts.")
            trial_scores.append((trial, precision, average_precision, learned_queries))
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))

    print(f"Same held-out test split: {len(test_embeddings)} embeddings, {random_queries} queries, {args.bits} bits.")
    print(f"Random projection (seed {args.projection_seed}): precision@{args.top_k}={random_precision:.4f}, AP={random_ap:.4f}.")
    print(f"ITQ (trained on {len(train_embeddings)} training embeddings): precision@{args.top_k}={itq_precision:.4f}, AP={itq_ap:.4f}.")
    print(f"Float ArcFace cosine (upper bound): precision@{args.top_k}={float_precision:.4f}, AP={float_ap:.4f}.")
    if args.top_trials is not None:
        print(f"Top {len(trial_scores)} trial models ranked by validation precision@10, then validation AP:")
        for rank, (trial, precision, average_precision, _) in enumerate(trial_scores, start=1):
            print(f"{rank:>2}. trial={trial['trial']}, seed={trial['model_seed']}, epochs={trial['epochs']}, validation precision@10={trial['validation_precision_at_10']:.4f}, validation AP={trial['validation_average_precision']:.4f}, test precision@{args.top_k}={precision:.4f}, test AP={average_precision:.4f}")
    else:
        _, learned_precision, learned_ap, _ = trial_scores[0]
        print(f"Learned hash: precision@{args.top_k}={learned_precision:.4f}, AP={learned_ap:.4f}.")


if __name__ == "__main__":
    main()
