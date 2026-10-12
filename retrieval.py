import heapq
import logging

import numpy as np

from ArcFace import extract_arcface_embedding
from caching import cache_embeddings, get_cached_embedding
from dataset_reader import iter_casia_index, load_casia_record


CACHE_BATCH_SIZE = 512
logger = logging.getLogger(__name__)


def cosine_similarity(first_embedding, second_embedding):
    first = np.asarray(first_embedding, dtype=np.float32).reshape(-1)
    second = np.asarray(second_embedding, dtype=np.float32).reshape(-1)

    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)

    similarity = float(np.dot(first, second) / (first_norm * second_norm))
    return float(np.clip(similarity, -1.0, 1.0))


def find_record_offset(dataset_directory, record_key):
    for key, offset in iter_casia_index(dataset_directory):
        if key == record_key:
            return offset
    raise ValueError(f"CASIA-WebFace record key {record_key} is not in train.idx.")


def rank_casia_records(model, dataset_directory, query_key, query_embedding, connection, signature, top_k, limit=None):
    matches = []
    pending_embeddings = []
    candidate_count = 0

    for record_key, record_offset in iter_casia_index(dataset_directory):
        if record_key == query_key:
            continue
        if limit is not None and candidate_count >= limit:
            break
        candidate_count += 1

        cached = get_cached_embedding(connection, signature, record_key)
        if cached is None:
            try:
                image, identity_label = load_casia_record(dataset_directory, record_key, record_offset)
                embedding = extract_arcface_embedding(model, image)
            except (OSError, ValueError) as error:
                logger.warning("Skipping CASIA record %s: %s", record_key, error)
                continue
            pending_embeddings.append((record_key, identity_label, embedding))
        else:
            identity_label, embedding = cached

        similarity = cosine_similarity(query_embedding, embedding)
        candidate = (similarity, -record_key, identity_label)
        if len(matches) < top_k:
            heapq.heappush(matches, candidate)
        elif candidate > matches[0]:
            heapq.heapreplace(matches, candidate)

        if len(pending_embeddings) >= CACHE_BATCH_SIZE:
            cache_embeddings(connection, signature, pending_embeddings)

        if candidate_count % 1000 == 0:
            logger.info("Scanned %s CASIA records", candidate_count)

    if pending_embeddings:
        cache_embeddings(connection, signature, pending_embeddings)

    return [(-negative_key, identity_label, similarity) for similarity, negative_key, identity_label in sorted(matches, reverse=True)]
