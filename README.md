# Face search with ArcFace and hashing

This project searches CASIA-WebFace for faces similar to a selected dataset
record. It uses ArcFace to make face embeddings, then can compare them either
with cosine similarity or with a basic binary-hashing version.

## Setup

Install Python, then install the packages used by the scripts:

```powershell
python -m pip install numpy pillow insightface onnxruntime
```

Put the CASIA-WebFace RecordIO files in `data\casia-webface`. The folder
should contain at least:

```text
data/
  casia-webface/
    train.idx
    train.rec
```

The dataset files are not included in this repository. By default, the scripts
look for them in that location. Use `--dataset` if your files are somewhere
else.

## Search with ArcFace

Start with a small run to check that the dataset and model work:

```powershell
python .\face_comparison.py --query-key 1 --top-k 10 --limit 1000
```

This compares record `1` with up to 1,000 other records and prints the closest
matches. Remove `--limit 1000` to search the full dataset; that can take a long
time the first time it runs.

The script saves embeddings in `data\arcface_embeddings.sqlite3` by default,
so later searches can reuse already processed records. Choose a different
record with `--query-key`, or change how many matches are printed with
`--top-k`. Run `python .\face_comparison.py --help` to see all options.

## Try binary hashing

Run ArcFace search first so there are embeddings in the cache. Then build and
search 256-bit codes:

```powershell
python .\binary_hashing.py --query-key 1 --bits 256 --top-k 10
```

This version uses a fixed random projection as a simple baseline. The output
includes precision@k and average precision for the selected query. Try a
different code length with `--bits`, or set a repeatable projection with
`--seed`. Run `python .\binary_hashing.py --help` for the other options.

## Load or resize an image

`image_input.py` can load a regular image and optionally save a resized copy:

```powershell
python .\image_input.py .\my-image.jpg --resize 224 224 --output .\resized.jpg
```

Without an image path, it loads the first CASIA-WebFace record instead. This
utility currently prepares images only; searches use a CASIA record key as
their query.
