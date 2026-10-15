from io import BytesIO
from pathlib import Path
import struct

from PIL import Image


DEFAULT_CASIA_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "casia-webface"
RECORDIO_MAGIC = 0xCED7230A
RECORDIO_HEADER_FORMAT = "<IfQQ"
RECORDIO_HEADER_SIZE = struct.calcsize(RECORDIO_HEADER_FORMAT)
RECORDIO_FRAMING_FORMAT = "<II"
RECORDIO_FRAMING_SIZE = struct.calcsize(RECORDIO_FRAMING_FORMAT)


def iter_casia_index(dataset_directory):
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
                raise ValueError(f"Invalid RecordIO index entry at {index_path}:{line_number}: {entry!r}")
            try:
                yield int(fields[0]), int(fields[1])
            except ValueError as error:
                raise ValueError(f"Invalid RecordIO key or offset at {index_path}:{line_number}: {entry!r}") from error


def load_casia_record(dataset_directory, record_key, record_offset):
    dataset_path = Path(dataset_directory).expanduser().resolve()
    record_path = dataset_path / "train.rec"
    if not record_path.is_file():
        raise FileNotFoundError(f"CASIA-WebFace records not found: {record_path}")

    with record_path.open("rb") as record_file:
        record_file.seek(record_offset)
        framing = record_file.read(RECORDIO_FRAMING_SIZE)
        if len(framing) != RECORDIO_FRAMING_SIZE:
            raise ValueError(f"RecordIO header is incomplete at offset {record_offset}.")

        magic, payload_length = struct.unpack(RECORDIO_FRAMING_FORMAT, framing)
        if magic != RECORDIO_MAGIC:
            raise ValueError(f"Invalid RecordIO signature at offset {record_offset}: 0x{magic:08x}.")
        if payload_length < RECORDIO_HEADER_SIZE:
            raise ValueError(f"RecordIO record at offset {record_offset} is too short.")

        payload = record_file.read(payload_length)
    if len(payload) != payload_length:
        raise ValueError(f"RecordIO record at offset {record_offset} is truncated.")

    _, label, embedded_key, _ = struct.unpack(RECORDIO_HEADER_FORMAT, payload[:RECORDIO_HEADER_SIZE])
    if embedded_key != record_key:
        raise ValueError(f"RecordIO index key {record_key} does not match record key {embedded_key}.")

    try:
        with Image.open(BytesIO(payload[RECORDIO_HEADER_SIZE:])) as source:
            source.load()
            image = source.convert("RGB")
    except OSError as error:
        raise ValueError(f"Could not decode CASIA record {record_key}: {error}") from error

    return image, float(label)
