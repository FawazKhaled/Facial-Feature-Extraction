# Face search with ArcFace and hashing

This project searches CASIA-WebFace for faces similar to a selected dataset
record. ArcFace turns each face into an embedding, which the search can compare
directly or convert into a basic binary code.

## Setup

Install Python, then install the packages used by the scripts:

```powershell
python -m pip install numpy pillow insightface onnxruntime torch
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

## Train the learned hashing model

First, run `face_comparison.py` to cache embeddings for the records you want to
use. Then train and evaluate the hashing model:

```powershell
python .\hashing_cnn.py --epochs 30 --bits 256
```

The script splits the cached images within each identity into training and
evaluation sets. It reports retrieval precision@10 and average precision on
the evaluation set, saves the model to `data\hashing_model.pt`, and stores the
packed binary codes in `data\binary_codes.sqlite3`. The floating-point
embeddings remain in the ArcFace cache because they are the model's training
input.

To compare the learned codes with the random-projection baseline on the same
held-out images, run:

```powershell
python .\compare_hashing.py
```

Use the same `--eval-ratio` and `--split-seed` values that were used when
training. The script reports precision@10 and average precision for both
methods.

## Load or resize an image

`image_input.py` can load a regular image and optionally save a resized copy:

```powershell
python .\image_input.py .\my-image.jpg --resize 224 224 --output .\resized.jpg
```

Without an image path, it loads the first CASIA-WebFace record instead. This
utility currently prepares images only; searches use a CASIA record key as
their query.

## What works now

The current workflow reads faces from CASIA-WebFace, gets ArcFace embeddings,
and saves them in a local SQLite cache. It can search for similar records with
cosine similarity. It can also make 256-bit codes with a fixed random
projection and compare those codes using Hamming distance.

The random projection is a baseline. A separate learned hashing model can now
be trained and its codes saved, but it is not yet connected to the search
commands. `image_input.py` can load a regular image, but the search commands
currently use a CASIA record key as the query.

## What still needs to be built

The trained model still needs to be compared with the random-projection
baseline across the same evaluation queries. After that, a live face search
would need to capture webcam frames, detect faces, create embeddings and hash
codes, and compare them with codes saved for registered people. It should also
be able to show "unknown" when a match is not reliable. Those enrollment and
webcam features are not in the project yet.
