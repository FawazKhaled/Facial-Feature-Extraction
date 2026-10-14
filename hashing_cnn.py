import numpy as np
from caching import load_embeddings

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
