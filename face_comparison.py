from pathlib import Path
from typing import Iterable, Iterator
import argparse

from PIL import Image

from dataset_loader import list_files
from image_input import load_image


IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "bmp", "webp")


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
    """Open dataset images one at a time to avoid loading the full dataset into memory."""
    for image_path in image_paths:
        with Image.open(image_path) as image:
            yield image_path, image.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a query face and dataset images for comparison."
    )
    parser.add_argument("query_image", help="Path to the face image to search for")
    parser.add_argument(
        "--dataset",
        default="data",
        help="Folder containing dataset images (default: data)",
    )
    args = parser.parse_args()

    try:
        query_image, query_path, dataset_image_paths = load_comparison_inputs(
            args.query_image,
            args.dataset,
        )
        print(f"Loaded query image: {query_path} ({query_image.size[0]}x{query_image.size[1]})")
        print(f"Found {len(dataset_image_paths)} dataset images in: {Path(args.dataset).resolve()}")
        print("Dataset images will be opened one at a time during comparison.")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
