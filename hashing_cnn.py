import numpy as np
import argparse
from pathlib import Path
import sqlite3

from caching import dataset_signature, load_embeddings, store_learned_codes
from dataset_reader import DEFAULT_CASIA_DIRECTORY
import torch


def build_hashing_model(dim=512, bit_length=256):
    import torch.nn as nn

    return nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Linear(512, bit_length), nn.Tanh())


# this is for getting the images we want to use for training and evaluation
def split_cached_embeddings(cache_path, signature, eval_ratio=0.2, seed=42):
    if not 0 < eval_ratio < 1:
        raise ValueError("eval_ratio must be between 0 and 1.")

    records = load_embeddings(cache_path, signature)
    if not records:
        raise ValueError("No cached embeddings found for this dataset.")

    embeddings = np.stack([embedding for _, _, embedding in records]).astype(np.float32, copy=False)
    labels = np.asarray([label for _, label, _ in records])
    rng = np.random.default_rng(seed)
    train_indexes = []
    eval_indexes = []

    for identity in np.unique(labels):
        identity_indexes = np.flatnonzero(labels == identity)
        rng.shuffle(identity_indexes)
        if len(identity_indexes) == 1:
            train_indexes.extend(identity_indexes)
            continue

        eval_count = min(max(round(len(identity_indexes) * eval_ratio), 1), len(identity_indexes) - 1)
        eval_indexes.extend(identity_indexes[:eval_count])
        train_indexes.extend(identity_indexes[eval_count:])

    rng.shuffle(train_indexes)
    rng.shuffle(eval_indexes)
    return embeddings[train_indexes], labels[train_indexes], embeddings[eval_indexes], labels[eval_indexes]


def train_hashing_model(model, embeddings, labels, epochs=30, batch_size=128, learning_rate=0.001, device=None):
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, TensorDataset

    train_data = TensorDataset(torch.as_tensor(embeddings, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.float32))
    batches = DataLoader(train_data, batch_size=batch_size, shuffle=True)
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
    parser = argparse.ArgumentParser(description="Train and evaluate a learned binary hashing model.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CASIA_DIRECTORY, help="CASIA-WebFace folder used to identify the embedding cache")
    parser.add_argument("--cache", type=Path, help="ArcFace embedding cache (default: data/arcface_embeddings.sqlite3)")
    parser.add_argument("--bits-cache", type=Path, help="Database for learned binary codes (default: data/binary_codes.sqlite3)")
    parser.add_argument("--model", type=Path, help="Where to save the trained model (default: data/hashing_model.pt)")
    parser.add_argument("--bits", type=int, default=256, help="Number of bits in each code")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--eval-ratio", type=float, default=0.2, help="Fraction of each identity held out for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Seed used to make the training and evaluation split")
    parser.add_argument("--device", help="PyTorch device, such as cpu or cuda")
    args = parser.parse_args()

    dataset_directory = args.dataset.expanduser().resolve()
    cache_path = args.cache or dataset_directory.parent / "arcface_embeddings.sqlite3"
    bits_cache_path = args.bits_cache or dataset_directory.parent / "binary_codes.sqlite3"
    model_path = args.model or dataset_directory.parent / "hashing_model.pt"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    try:
        if args.bits < 1 or args.epochs < 1 or args.batch_size < 2:
            raise ValueError("bits and epochs must be positive, and batch-size must be at least 2.")
        signature = dataset_signature(dataset_directory)
        train_embeddings, train_labels, eval_embeddings, eval_labels = split_cached_embeddings(cache_path, signature, args.eval_ratio, args.seed)
        model = build_hashing_model(train_embeddings.shape[1], args.bits)
        model, history = train_hashing_model(model, train_embeddings, train_labels, epochs=args.epochs, batch_size=args.batch_size, device=device)
        precision_at_k, average_precision, evaluated_queries = evaluate_hashing_model(model, eval_embeddings, eval_labels, device=device)
        records = load_embeddings(cache_path, signature)
        all_embeddings = np.stack([embedding for _, _, embedding in records])
        all_codes = encode_hash(model, all_embeddings, device=device)
        stored_codes = [(record_key, label, code) for (record_key, label, _), code in zip(records, all_codes)]
        store_learned_codes(bits_cache_path, signature, stored_codes, args.bits)
        save_hashing_model(model, model_path)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))

    print(f"Trained for {len(history)} epochs on {len(train_embeddings)} embeddings using {device}.")
    print(f"Evaluation on {evaluated_queries} queries: precision@10={precision_at_k:.4f}, AP={average_precision:.4f}.")
    print(f"Stored {len(stored_codes)} learned binary codes in {bits_cache_path}.")
    print(f"Saved hashing model to {model_path}.")


if __name__ == "__main__":
    main()
