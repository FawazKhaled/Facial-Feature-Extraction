from pathlib import Path
from collections.abc import Iterable, Iterator, Sequence
from importlib import import_module
import logging
from typing import Protocol, cast
import argparse

import numpy as np
from PIL import Image

from dataset_loader import list_files
from image_input import load_image


IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "bmp", "webp")
EMBEDDING_DIMENSION = 512
logger = logging.getLogger(__name__)


class DetectedFace(Protocol):
    embedding: np.ndarray


class ArcFaceModel(Protocol):
    def prepare(self, ctx_id: int = ..., det_size: tuple[int, int] = ...) -> None: ...

    def get(self, img: np.ndarray) -> Sequence[DetectedFace]: ...


def load_comparison_inputs(
    query_image_path: str,
    dataset_directory: str,
) -> tuple[Image.Image, Path, list[Path]]:
    # Load the query face and find supported image files in the dataset.
    query_image, query_path = load_image(query_image_path)

    dataset_path = Path(dataset_directory).expanduser().resolve()
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Dataset folder not found: {dataset_path}")

    dataset_image_paths = list_files(
        dataset_path,
        extensions=IMAGE_EXTENSIONS,
    )
    if not dataset_image_paths:
        raise ValueError(f"No supported images found in: {dataset_path}")

    return query_image, query_path, dataset_image_paths


def iter_dataset_images(
    image_paths: Iterable[Path],
) -> Iterator[tuple[Path, Image.Image]]:
    # Open dataset images one at a time to avoid loading the full dataset into memory.
    for image_path in image_paths:
        with Image.open(image_path) as image:
            yield image_path, image.convert("RGB")


def load_arcface_model() -> ArcFaceModel:
    # Load InsightFace's pretrained ArcFace model using CPU inference.
    try:
        insightface_app = import_module("insightface.app")
    except ImportError as error:
        raise RuntimeError(
            "ArcFace extraction requires InsightFace and ONNX Runtime. "
            "Install them with: pip install insightface onnxruntime"
        ) from error

    face_analysis = getattr(insightface_app, "FaceAnalysis")
    model = face_analysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
    model.prepare(ctx_id=-1)
    return cast(ArcFaceModel, model)


def extract_arcface_embedding(
    model: ArcFaceModel,
    image: Image.Image,
) -> np.ndarray:
    # Detect one face and return its normalized 512-D ArcFace feature vector.
    rgb_image = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr_image = rgb_image[:, :, ::-1].copy()
    faces = model.get(bgr_image)

    if len(faces) != 1:
        raise ValueError(
            f"Expected exactly one face in the image, but detected {len(faces)}."
        )

    embedding = np.asarray(faces[0].embedding, dtype=np.float32).reshape(-1)
    if embedding.size != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected a {EMBEDDING_DIMENSION}-D ArcFace embedding, "
            f"received {embedding.size} values."
        )

    norm = np.linalg.norm(embedding)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("ArcFace returned an invalid or zero-length embedding.")

    return embedding / norm


def iter_dataset_embeddings(
    model: ArcFaceModel,
    image_paths: Iterable[Path],
) -> Iterator[tuple[Path, np.ndarray]]:
    # Extract dataset embeddings one image at a time.
    for image_path, image in iter_dataset_images(image_paths):
        yield image_path, extract_arcface_embedding(model, image)


def cosine_similarity(
    first_embedding: np.ndarray,
    second_embedding: np.ndarray,
) -> float:
    # Return the cosine similarity between two 512-D face embeddings.
    first = np.asarray(first_embedding, dtype=np.float32).reshape(-1)
    second = np.asarray(second_embedding, dtype=np.float32).reshape(-1)

    if first.size != EMBEDDING_DIMENSION or second.size != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Cosine similarity requires two {EMBEDDING_DIMENSION}-D embeddings."
        )
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Embeddings must contain only finite values.")

    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm == 0 or second_norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero-length embedding.")

    similarity = float(np.dot(first, second) / (first_norm * second_norm))
    return float(np.clip(similarity, -1.0, 1.0))


def iter_dataset_similarities(
    model: ArcFaceModel,
    query_embedding: np.ndarray,
    image_paths: Iterable[Path],
) -> Iterator[tuple[Path, float]]:
    # Compare a query embedding with each readable, single-face dataset image.
    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                dataset_embedding = extract_arcface_embedding(
                    model,
                    image.convert("RGB"),
                )
            yield image_path, cosine_similarity(query_embedding, dataset_embedding)
        except (OSError, ValueError) as error:
            logger.warning("Skipping %s: %s", image_path, error)


def rank_similarities(
    similarities: Iterable[tuple[Path, float]],
    top_k: int | None = None,
) -> list[tuple[Path, float]]:
    # Sort matches from most to least similar, optionally keeping only top_k.
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    ranked = sorted(similarities, key=lambda result: result[1], reverse=True)
    return ranked if top_k is None else ranked[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a query face with dataset images using ArcFace cosine similarity."
    )
    parser.add_argument("query_image", help="Path to the face image to search for")
    parser.add_argument(
        "--dataset",
        default="data",
        help="Folder containing dataset images (default: data)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of highest-similarity matches to display (default: 10)",
    )
    args = parser.parse_args()

    try:
        if args.top_k < 1:
            parser.error("--top-k must be at least 1.")

        query_image, query_path, dataset_image_paths = load_comparison_inputs(
            args.query_image,
            args.dataset,
        )
        model = load_arcface_model()
        query_embedding = extract_arcface_embedding(model, query_image)
        print(f"Loaded query image: {query_path} ({query_image.size[0]}x{query_image.size[1]})")
        print(f"Found {len(dataset_image_paths)} dataset images in: {Path(args.dataset).resolve()}")
        ranked_matches = rank_similarities(
            iter_dataset_similarities(model, query_embedding, dataset_image_paths),
            top_k=args.top_k,
        )
        if not ranked_matches:
            parser.error("No dataset images could be compared successfully.")

        print(f"Top {len(ranked_matches)} matches (cosine similarity, highest first):")
        for rank, (image_path, similarity) in enumerate(ranked_matches, start=1):
            print(f"{rank:>3}. {similarity:.4f}\t{image_path}")
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
