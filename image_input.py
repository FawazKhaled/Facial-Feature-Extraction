from pathlib import Path
from io import BytesIO
from collections.abc import Iterator
import struct
import argparse

from PIL import Image


DEFAULT_CASIA_DIRECTORY = Path(__file__).resolve().parent / "data" / "casia-webface"
RECORDIO_MAGIC = 0xCED7230A
RECORDIO_HEADER_FORMAT = "<IfQQ"
RECORDIO_HEADER_SIZE = struct.calcsize(RECORDIO_HEADER_FORMAT)
RECORDIO_FRAMING_FORMAT = "<II"
RECORDIO_FRAMING_SIZE = struct.calcsize(RECORDIO_FRAMING_FORMAT)


def load_image(image_path: str):
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as source:
        image = source.convert("RGB")
    return image, path


def iter_casia_index(dataset_directory: str | Path = DEFAULT_CASIA_DIRECTORY) -> Iterator[tuple[int, int]]:
    dataset_path = Path(dataset_directory).expanduser().resolve()
    index_path = dataset_path / "train.idx"

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"CASIA-WebFace folder not found: {dataset_path}")
    if not index_path.is_file():
        raise FileNotFoundError(f"CASIA-WebFace index not found: {index_path}")

    with index_path.open("r", encoding="ascii") as index_file:
        for line_number, line in enumerate(index_file, start=1):
            entry = line.strip()
            if not entry:
                continue
            fields = entry.split("\t")
            if len(fields) != 2:
                raise ValueError(
                    f"Invalid RecordIO index entry at {index_path}:{line_number}: {entry!r}"
                )
            try:
                yield int(fields[0]), int(fields[1])
            except ValueError as error:
                raise ValueError(
                    f"Invalid RecordIO key or offset at {index_path}:{line_number}: {entry!r}"
                ) from error


def load_casia_record(dataset_directory: str | Path, record_key: int, record_offset: int) -> tuple[Image.Image, float]:
    dataset_path = Path(dataset_directory).expanduser().resolve()
    record_path = dataset_path / "train.rec"
    if not record_path.is_file():
        raise FileNotFoundError(f"CASIA-WebFace records not found: {record_path}")

    with record_path.open("rb") as record_file:
        record_file.seek(record_offset)
        # Each indexed record starts with the RecordIO magic value and payload size.
        framing = record_file.read(RECORDIO_FRAMING_SIZE)
        if len(framing) != RECORDIO_FRAMING_SIZE:
            raise ValueError(f"RecordIO header is incomplete at offset {record_offset}.")

        magic, payload_length = struct.unpack(RECORDIO_FRAMING_FORMAT, framing)
        if magic != RECORDIO_MAGIC:
            raise ValueError(
                f"Invalid RecordIO signature at offset {record_offset}: 0x{magic:08x}."
            )
        if payload_length < RECORDIO_HEADER_SIZE:
            raise ValueError(f"RecordIO record at offset {record_offset} is too short.")

        payload = record_file.read(payload_length)
    if len(payload) != payload_length:
        raise ValueError(f"RecordIO record at offset {record_offset} is truncated.")

    # The header carries the identity label and key; the remaining bytes are JPEG data.
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
            f"Could not decode CASIA record {record_key}: {error}"
        ) from error

    return image, float(label)


def load_first_casia_image(dataset_directory: str | Path = DEFAULT_CASIA_DIRECTORY) -> tuple[Image.Image, int, float]:
    first_record = next(iter_casia_index(dataset_directory), None)
    if first_record is None:
        raise ValueError(
            f"CASIA-WebFace index is empty: "
            f"{Path(dataset_directory).expanduser().resolve() / 'train.idx'}"
        )
    record_key, record_offset = first_record
    image, label = load_casia_record(dataset_directory, record_key, record_offset)
    return image, record_key, label


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
