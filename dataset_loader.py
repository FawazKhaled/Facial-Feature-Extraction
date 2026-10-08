from pathlib import Path
import argparse


def list_files(root_dir, extensions=None, recursive=True):
    # Return a list of files under root_dir, optionally filtered by extension.
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    if extensions is None:
        extensions = []
    else:
        extensions = [ext.lower().lstrip('.') for ext in extensions]

    files = []
    iterator = root.rglob('*') if recursive else root.iterdir()

    for item in iterator:
        if item.is_file():
            if not extensions:
                files.append(item)
            else:
                ext = item.suffix.lower().lstrip('.')
                if ext in extensions:
                    files.append(item)

    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="List files in a dataset folder.")
    parser.add_argument("folder", nargs="?", default="data", help="Folder to scan")
    parser.add_argument("--ext", nargs="*", default=None, help="Filter by extension(s), e.g. jpg png")
    parser.add_argument("--no-recursive", action="store_true", help="Only scan the top level")
    args = parser.parse_args()

    try:
        files = list_files(args.folder, extensions=args.ext, recursive=not args.no_recursive)
        print(f"Found {len(files)} files in: {args.folder}")
        for file in files[:20]:
            print(file)
        if len(files) > 20:
            print("...")
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
