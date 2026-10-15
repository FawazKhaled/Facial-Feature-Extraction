import numpy as np
import argparse
import json
from pathlib import Path
import random
import sqlite3

from lib.caching import dataset_signature, load_embeddings, store_learned_codes
from lib.dataset_reader import DEFAULT_CASIA_DIRECTORY
import torch


def build_hashing_model(dim=512, bit_length=256):
    import torch.nn as nn

    return nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Linear(512, bit_length), nn.Tanh())


# this is for getting the images we want to use for training and evaluation
def split_cached_embeddings(cache_path, signature, validation_ratio=0.2, test_ratio=0.2, seed=42):
    if validation_ratio <= 0 or test_ratio <= 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio and test_ratio must be positive and add up to less than 1.")

    records = load_embeddings(cache_path, signature)
    if not records:
        raise ValueError("No cached embeddings found for this dataset.")

    embeddings = np.stack([embedding for _, _, embedding in records]).astype(np.float32, copy=False)
    labels = np.asarray([label for _, label, _ in records])
    rng = np.random.default_rng(seed)
    train_indexes = []
    validation_indexes = []
    test_indexes = []

    for identity in np.unique(labels):
        identity_indexes = np.flatnonzero(labels == identity)
        rng.shuffle(identity_indexes)
        sample_count = len(identity_indexes)
        if sample_count == 1:
            train_indexes.extend(identity_indexes)
            continue

        validation_count = max(round(sample_count * validation_ratio), 1)
        test_count = max(round(sample_count * test_ratio), 1) if sample_count >= 3 else 0
        while validation_count + test_count >= sample_count:
            if validation_count >= test_count and validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                test_count = 0
                break

        validation_indexes.extend(identity_indexes[:validation_count])
        test_indexes.extend(identity_indexes[validation_count:validation_count + test_count])
        train_indexes.extend(identity_indexes[validation_count + test_count:])

    rng.shuffle(train_indexes)
    rng.shuffle(validation_indexes)
    rng.shuffle(test_indexes)
    return (embeddings[train_indexes], labels[train_indexes], embeddings[validation_indexes], labels[validation_indexes],
            embeddings[test_indexes], labels[test_indexes])


def train_hashing_model(model, embeddings, labels, epochs=30, batch_size=128, learning_rate=0.001, device=None, seed=42):
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, TensorDataset

    train_data = TensorDataset(torch.as_tensor(embeddings, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.float32))
    batches = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_embeddings, batch_labels in batches:
            if len(batch_labels) < 2:
                continue
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)

            codes = model(batch_embeddings)
            similarities = torch.matmul(codes, codes.T) / codes.shape[1]
            same_identity = batch_labels[:, None] == batch_labels[None, :]
            pair_mask = ~torch.eye(len(batch_labels), dtype=torch.bool, device=device)
            pair_targets = same_identity[pair_mask].float()
            positive_pairs = pair_targets.sum()
            negative_pairs = len(pair_targets) - positive_pairs
            pair_logits = similarities[pair_mask] * 8
            pair_weight = negative_pairs / positive_pairs if positive_pairs > 0 and negative_pairs > 0 else torch.ones((), device=device)
            pair_loss = functional.binary_cross_entropy_with_logits(pair_logits, pair_targets, pos_weight=pair_weight)
            quantization_loss = torch.mean((torch.abs(codes) - 1) ** 2)
            balance_loss = torch.mean(torch.mean(codes, dim=0) ** 2)
            loss = pair_loss + 0.1 * quantization_loss + 0.01 * balance_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1

        history.append(epoch_loss / batch_count if batch_count else 0.0)

    return model, history


def encode_hash(model, embeddings, device=None):

    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        codes = model(torch.as_tensor(values, dtype=torch.float32, device=device)).cpu().numpy() >= 0

    packed_codes = np.packbits(codes, axis=1, bitorder="little")
    return [code.tobytes() for code in packed_codes]


def save_hashing_model(model, model_path):

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_dim": model[0].in_features, "bit_length": model[2].out_features, "state_dict": model.state_dict()}, model_path)


def load_hashing_model(model_path, device="cpu"):

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = build_hashing_model(checkpoint["input_dim"], checkpoint["bit_length"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_hashing_model(model, embeddings, labels, top_k=10, device=None):
    return evaluate_binary_codes(encode_hash(model, embeddings, device=device), labels, top_k)


def evaluate_binary_codes(codes, labels, top_k=10):
    packed_codes = np.frombuffer(b"".join(codes), dtype=np.uint8).reshape(len(codes), -1)
    labels = np.asarray(labels)
    precisions = []
    average_precisions = []

    for query_index, query_label in enumerate(labels):
        other_indexes = np.arange(len(labels)) != query_index
        distances = np.unpackbits(np.bitwise_xor(packed_codes[other_indexes], packed_codes[query_index]), axis=1).sum(axis=1)
        order = np.lexsort((np.flatnonzero(other_indexes), distances))
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


def main():
    parser = argparse.ArgumentParser(description="Train multiple hashing models with randomized seeds and epoch counts.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CASIA_DIRECTORY, help="CASIA-WebFace folder used to identify the embedding cache")
    parser.add_argument("--cache", type=Path, help="ArcFace embedding cache (default: data/arcface_embeddings.sqlite3)")
    parser.add_argument("--bits-cache", type=Path, help="Database for learned binary codes (default: data/binary_codes.sqlite3)")
    parser.add_argument("--model", type=Path, help="Where to save the trained model (default: data/hashing_model.pt)")
    parser.add_argument("--trials-dir", type=Path, help="Where to save trial models and results (default: data/hashing_trials)")
    parser.add_argument("--trials", type=int, default=10, help="Number of randomized training trials")
    parser.add_argument("--bits", type=int, default=256, help="Number of bits in each code")
    parser.add_argument("--epochs", type=int, help="Use this fixed epoch count for every trial instead of randomizing it")
    parser.add_argument("--min-epochs", type=int, default=10, help="Minimum randomized epoch count")
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum randomized epoch count")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--validation-ratio", type=float, default=0.2, help="Fraction of each identity held out for validation")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of each identity held out for final testing")
    parser.add_argument("--split-seed", "--seed", dest="split_seed", type=int, default=42, help="Fixed seed used for the shared training/validation/test split")
    parser.add_argument("--search-seed", type=int, default=42, help="Seed controlling randomized trial settings")
    parser.add_argument("--device", help="PyTorch device, such as cpu or cuda")
    args = parser.parse_args()
    if args.epochs is not None:
        args.min_epochs = args.epochs
        args.max_epochs = args.epochs

    dataset_directory = args.dataset.expanduser().resolve()
    cache_path = args.cache or dataset_directory.parent / "arcface_embeddings.sqlite3"
    bits_cache_path = args.bits_cache or dataset_directory.parent / "binary_codes.sqlite3"
    model_path = args.model or dataset_directory.parent / "hashing_model.pt"
    trials_dir = args.trials_dir or dataset_directory.parent / "hashing_trials"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    try:
        if args.trials < 1 or args.bits < 1 or args.min_epochs < 1 or args.max_epochs < args.min_epochs or args.batch_size < 2:
            raise ValueError("trials and bits must be positive, epoch range must be valid, and batch-size must be at least 2.")
        if args.validation_ratio <= 0 or args.test_ratio <= 0 or args.validation_ratio + args.test_ratio >= 1:
            raise ValueError("validation-ratio and test-ratio must be positive and add up to less than 1.")
        signature = dataset_signature(dataset_directory)
        train_embeddings, train_labels, validation_embeddings, validation_labels, _, _ = split_cached_embeddings(cache_path, signature, args.validation_ratio, args.test_ratio, args.split_seed)
        trials_dir.mkdir(parents=True, exist_ok=True)
        setting_rng = random.Random(args.search_seed)
        trial_results = []

        for trial_number in range(1, args.trials + 1):
            model_seed = setting_rng.randrange(0, 2**31)
            epochs = setting_rng.randint(args.min_epochs, args.max_epochs)
            torch.manual_seed(model_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(model_seed)
            model = build_hashing_model(train_embeddings.shape[1], args.bits)
            model, _ = train_hashing_model(model, train_embeddings, train_labels, epochs=epochs, batch_size=args.batch_size, device=device, seed=model_seed)
            precision, average_precision, evaluated_queries = evaluate_hashing_model(model, validation_embeddings, validation_labels, device=device)
            trial_model_path = trials_dir / f"trial_{trial_number:03d}_seed_{model_seed}.pt"
            save_hashing_model(model, trial_model_path)
            trial_results.append({"trial": trial_number, "model_seed": model_seed, "epochs": epochs, "validation_precision_at_10": precision,
                                  "validation_average_precision": average_precision, "validation_queries": evaluated_queries,
                                  "model_path": str(trial_model_path.resolve())})
            print(f"Trial {trial_number}/{args.trials}: seed={model_seed}, epochs={epochs}, validation precision@10={precision:.4f}, AP={average_precision:.4f}.")

        trial_results.sort(key=lambda trial: (trial["validation_precision_at_10"], trial["validation_average_precision"]), reverse=True)
        best_model = load_hashing_model(trial_results[0]["model_path"], device)
        save_hashing_model(best_model, model_path)
        records = load_embeddings(cache_path, signature)
        all_embeddings = np.stack([embedding for _, _, embedding in records])
        all_codes = encode_hash(best_model, all_embeddings, device=device)
        stored_codes = [(record_key, label, code) for (record_key, label, _), code in zip(records, all_codes)]
        store_learned_codes(bits_cache_path, signature, stored_codes, args.bits)
        manifest_path = trials_dir / "trials.json"
        manifest = {"dataset_signature": signature, "bits": args.bits, "validation_ratio": args.validation_ratio,
                    "test_ratio": args.test_ratio, "split_seed": args.split_seed, "search_seed": args.search_seed,
                    "batch_size": args.batch_size, "trials": trial_results}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))

    print(f"Finished {len(trial_results)} trials on {len(train_embeddings)} training and {len(validation_embeddings)} validation embeddings using {device}.")
    print("Top trials by validation precision@10, then AP:")
    for rank, trial in enumerate(trial_results[:10], start=1):
        print(f"{rank:>2}. trial={trial['trial']}, seed={trial['model_seed']}, epochs={trial['epochs']}, precision@10={trial['validation_precision_at_10']:.4f}, AP={trial['validation_average_precision']:.4f}")
    print(f"Stored {len(stored_codes)} learned binary codes in {bits_cache_path}.")
    print(f"Saved best validation model to {model_path} and trial details to {manifest_path}.")


if __name__ == "__main__":
    main()
