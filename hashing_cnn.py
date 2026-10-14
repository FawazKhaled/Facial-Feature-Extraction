import numpy as np
from pathlib import Path
from caching import load_embeddings
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
            pair_weight = negative_pairs / positive_pairs if positive_pairs > 0 and negative_pairs > 0 else 1.0
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
