import argparse
from pathlib import Path

from PIL import Image

from lib.dataset_reader import DEFAULT_CASIA_DIRECTORY, iter_casia_index, load_casia_record

def load_image(image_path):
    path = Path(image_path).expanduser().resolve()
    with Image.open(path) as source:
        image = source.convert("RGB")
    return (image, path)



def load_first_casia_image(dataset_directory):
    first_record = next(iter_casia_index(dataset_directory), None)
    if first_record is None:
        raise ValueError(f"CASIA-WebFace index is empty: {Path(dataset_directory).expanduser().resolve() / 'train.idx'}")
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
    parser = argparse.ArgumentParser(description="Load a CASIA-WebFace sample or a specified image for preprocessing.")
    parser.add_argument("image_path", nargs="?", help="Optional path to a specific image; defaults to the first CASIA-WebFace sample")
    parser.add_argument("--dataset", default=DEFAULT_CASIA_DIRECTORY, type=Path, help="CASIA-WebFace folder containing train.idx and train.rec")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(224, 224))
    parser.add_argument("--output", help="Optional output path for the resized image")
    args = parser.parse_args()

    try:
        if args.image_path:
            image, path = load_image(args.image_path)
            print(f"Loaded image: {path}")
        else:
            image, record_key, label = load_first_casia_image(args.dataset)
            print(f"Loaded first CASIA-WebFace sample: record {record_key}, identity label {label:g}")

        print(f"Original size: {image.size}")
        print(f"Mode: {image.mode}")

        resized = preprocess_image(image, target_size=tuple(args.resize), output_path=args.output)
        print(f"Resized size: {resized.size}")

    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
