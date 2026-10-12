import numpy as np


def create_matrix(embedding_dimension, bit_length, seed):
    generator = np.random.default_rng(seed)
    return generator.standard_normal(size=(embedding_dimension, bit_length), dtype=np.float32)


def encode_embedding(embedding, projection_matrix):
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    projected_values = vector @ projection_matrix
    bits = np.greater_equal(projected_values, 0)
    return np.packbits(bits, bitorder="little").tobytes()


def ham_dis(first_code, second_code):
    first = np.frombuffer(first_code, dtype=np.uint8)
    second = np.frombuffer(second_code, dtype=np.uint8)
    xor_values = np.bitwise_xor(first, second)
    return int(np.unpackbits(xor_values).sum())


def rank_hamming(codes, query_key, top_k):
    code_by_key = {record_key: (label, code) for record_key, label, code in codes}

    if query_key not in code_by_key:
        raise ValueError(f"Query key {query_key} not found in the provided codes.")
    query_label, query_code = code_by_key[query_key]
    ranked = sorted(((record_key, label, ham_dis(query_code, code)) for record_key, label, code in codes if record_key != query_key), key=lambda match: (match[2], match[0]))

    result_count = min(top_k, len(ranked))
    relevant_total = sum(label == query_label for _, label, _ in ranked)
    relevant_seen = 0
    precision_sum = 0.0
    for rank, (_, label, _) in enumerate(ranked, start=1):
        if label == query_label:
            relevant_seen += 1
            precision_sum += relevant_seen / rank

    precision_at_k = sum(label == query_label for _, label, _ in ranked[:result_count]) / result_count if result_count else 0.0
    average_precision = precision_sum / relevant_total if relevant_total else 0.0
    return ranked[:top_k], precision_at_k, average_precision
