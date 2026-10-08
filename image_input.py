from pathlib import Path
import argparse

from PIL import Image


def load_image(image_path: str):
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path)
    image = image.convert("RGB")
    return image, path


def preprocess_image(image, target_size=(224, 224), output_path=None):
    resized = image.resize(target_size)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        resized.save(output)
        print(f"Saved preprocessed image to: {output}")

    return resized


def main():
    parser = argparse.ArgumentParser(description="Load an image for face-model preprocessing.")
    parser.add_argument("image_path", nargs="?", help="Path to the input image file")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(224, 224))
    parser.add_argument("--output", help="Optional output path for the resized image")
    args = parser.parse_args()

    image_path = args.image_path
    if image_path is None:
        image_path = input("Enter the image path: ").strip().strip('"').strip("'").strip(" ")

    try:
        image, path = load_image(image_path)
        print(f"Loaded image: {path}")
        print(f"Original size: {image.size}")
        print(f"Mode: {image.mode}")

        resized = preprocess_image(image, target_size=tuple(args.resize), output_path=args.output)
        print(f"Resized size: {resized.size}")

    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
