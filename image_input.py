from pathlib import Path
from io import BytesIO
import struct
import argparse

from PIL import Image


DEFAULT_CASIA_DIRECTORY = Path(__file__).resolve().parent / "data" / "casia-webface"
RECORDIO_MAGIC = 0xCED7230A
RECORDIO_HEADER_FORMAT = "<IfQQ"
RECORDIO_HEADER_SIZE = struct.calcsize(RECORDIO_HEADER_FORMAT)


def load_image(image_path: str):
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as source:
        image = source.convert("RGB")
    return image, path


def load_first_casia_image(
    dataset_directory: str | Path = DEFAULT_CASIA_DIRECTORY,
) -> tuple[Image.Image, int, float]:
    """Load the first indexed face image from a CASIA-WebFace RecordIO dataset."""
    dataset_path = Path(dataset_directory).expanduser().resolve()
    index_path = dataset_path / "train.idx"
    record_path = dataset_path / "train.rec"

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"CASIA-WebFace folder not found: {dataset_path}")
    if not index_path.is_file() or not record_path.is_file():
        raise FileNotFoundError(
            f"Expected train.idx and train.rec in CASIA-WebFace folder: {dataset_path}"
        )

    with index_path.open("r", encoding="ascii") as index_file:
        first_entry = next((line.strip() for line in index_file if line.strip()), None)
    if first_entry is None:
        raise ValueError(f"CASIA-WebFace index is empty: {index_path}")

    index_fields = first_entry.split("\t")
    if len(index_fields) != 2:
        raise ValueError(f"Invalid RecordIO index entry in {index_path}: {first_entry!r}")
    try:
        record_key, record_offset = map(int, index_fields)
    except ValueError as error:
        raise ValueError(
            f"Invalid RecordIO key or offset in {index_path}: {first_entry!r}"
        ) from error

    with record_path.open("rb") as record_file:
        record_file.seek(record_offset)
        framing = record_file.read(8)
        if len(framing) != 8:
            raise ValueError(f"RecordIO header is incomplete at offset {record_offset}.")

        magic, payload_length = struct.unpack("<II", framing)
        if magic != RECORDIO_MAGIC:
            raise ValueError(
                f"Invalid RecordIO signature at offset {record_offset}: 0x{magic:08x}."
            )
        if payload_length < RECORDIO_HEADER_SIZE:
            raise ValueError(f"RecordIO record at offset {record_offset} is too short.")

        payload = record_file.read(payload_length)
    if len(payload) != payload_length:
        raise ValueError(f"RecordIO record at offset {record_offset} is truncated.")

    _, label, embedded_key, _ = struct.unpack(
        RECORDIO_HEADER_FORMAT,
        payload[:RECORDIO_HEADER_SIZE],
    )
    if embedded_key != record_key:
        raise ValueError(
            f"RecordIO index key {record_key} does not match record key {embedded_key}."
        )

    try:
        with Image.open(BytesIO(payload[RECORDIO_HEADER_SIZE:])) as source:
            source.load()
            image = source.convert("RGB")
    except OSError as error:
        raise ValueError(
            f"Could not decode the first image in CASIA-WebFace: {error}"
        ) from error

    return image, record_key, float(label)


def preprocess_image(image, target_size=(224, 224), output_path=None):
    resized = image.resize(target_size)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        resized.save(output)
        print(f"Saved preprocessed image to: {output}")

    return resized


def main():
    parser = argparse.ArgumentParser(
        description="Load a CASIA-WebFace sample or a specified image for preprocessing."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Optional path to a specific image; defaults to the first CASIA-WebFace sample",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_CASIA_DIRECTORY,
        type=Path,
        help="CASIA-WebFace folder containing train.idx and train.rec",
    )
    parser.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(224, 224))
    parser.add_argument("--output", help="Optional output path for the resized image")
    args = parser.parse_args()

    try:
        if args.image_path:
            image, path = load_image(args.image_path)
            print(f"Loaded image: {path}")
        else:
            image, record_key, label = load_first_casia_image(args.dataset)
            print(
                f"Loaded first CASIA-WebFace sample: record {record_key}, "
                f"identity label {label:g}"
            )

        print(f"Original size: {image.size}")
        print(f"Mode: {image.mode}")

        resized = preprocess_image(image, target_size=tuple(args.resize), output_path=args.output)
        print(f"Resized size: {resized.size}")

    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
