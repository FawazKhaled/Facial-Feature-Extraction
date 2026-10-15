# Face search with ArcFace and binary hashing

This project reads face images from CASIA-WebFace and turns each face into a
512-number ArcFace embedding. It then turns each embedding into a 256-bit
binary code. The codes are much smaller than the embeddings and can be compared
quickly by counting how many bits differ (Hamming distance).

There are three ways to make the codes:

- **Random projection:** multiplies the embedding by a fixed random matrix and
  keeps the sign of each value. This is the simplest baseline.
- **ITQ (Iterative Quantization):** fits a projection to the training
  embeddings without using identity labels. This is a stronger baseline.
- **Learned hashing model:** a small neural network trained on identity labels
  so that photos of the same person get similar codes.

## Project layout

```text
Research/
  face_comparison.py    Search with ArcFace embeddings and fill the embedding cache
  binary_hashing.py     Make random-projection codes and search with them
  hashing_cnn.py        Train the learned hashing model
  compare_hashing.py    Score every method on the same held-out test set
  image_input.py        Load or resize a single image
  dataset_loader.py     List the files in a folder
  lib/
    ArcFace.py          Loads the ArcFace model and extracts embeddings
    caching.py          SQLite storage for embeddings and binary codes
    dataset_reader.py   Reads images from the CASIA-WebFace RecordIO files
    hashing.py          Random projection and Hamming distance helpers
    retrieval.py        Cosine similarity search over the dataset
  data/                 Dataset, caches, and trained models (not in git)
```

Run every script from the `Research` folder. The files in `lib` are shared
helpers used by the scripts, so you don't run them directly.

## Setup

Install Python, then install the packages the scripts use:

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

The dataset isn't included in this repository. The scripts look for it in
that folder by default. If your copy is somewhere else, pass its location with
`--dataset`.

## Step 1: Extract ArcFace embeddings

Start with a small run to check that the dataset and the model work:

```powershell
python .\face_comparison.py --query-key 1 --top-k 10 --limit 1000
```

This compares record `1` with up to 1,000 other records and prints the closest
matches by cosine similarity. Every embedding it computes is saved in
`data\arcface_embeddings.sqlite3`, so later runs skip records that are already
done. Raise or remove `--limit` to process more of the dataset. The full
dataset takes a long time the first time.

All the hashing steps below work on whatever embeddings are in this cache, so
run this step first. Use `--query-key` to pick a different record and `--top-k`
to change how many matches are printed. Run
`python .\face_comparison.py --help` to see every option.

## Step 2: Try random-projection codes

```powershell
python .\binary_hashing.py --query-key 1 --bits 256 --top-k 10
```

This makes a code for every cached embedding, saves the codes, and prints the
closest matches to the query by Hamming distance. It also prints precision@k
(how many of the top k results are the same person) and average precision for
that query. Use `--bits` to change the code length and `--seed` to change the
random matrix.

## Step 3: Train the learned hashing model

```powershell
python .\hashing_cnn.py --trials 20 --bits 256
```

The cached embeddings are split per person into 60% training, 20% validation,
and 20% test (split seed 42 by default). Every trial uses that same split but
gets its own random training seed and a random epoch count between 10 and 50.
Use `--min-epochs` and `--max-epochs` to change that range, or `--epochs` to
give every trial the same count.

Trials are ranked by validation precision@10, with validation average
precision as the tie-breaker. When training finishes:

- each trial's model and the full results (`trials.json`) are in
  `data\hashing_trials`
- the best model is copied to `data\hashing_model.pt`
- the best model's codes for every cached embedding are saved in
  `data\binary_codes.sqlite3`

Running the script again with the default paths replaces those files. To keep
an earlier run, give the new one its own paths with `--trials-dir`, `--model`,
and `--bits-cache`. For example, this trains one long trial without touching
the defaults:

```powershell
python .\hashing_cnn.py --trials 1 --epochs 512 --trials-dir data\hashing_trials_e512 --model data\hashing_model_e512.pt --bits-cache data\binary_codes_e512.sqlite3
```

For a quick test, use something like `--trials 2 --epochs 1`.

## Step 4: Compare the methods

```powershell
python .\compare_hashing.py --top-trials 10
```

This scores everything on the same held-out test images:

- random projection
- ITQ, fitted on the training split only
- searching with the full float embeddings, which is the best any binary code
  can realistically do
- the top 10 trials from Step 3, in validation order

The script reads the split settings from `trials.json`, so its test set always
matches the one used during training. Pick models by their validation scores,
not their test scores, so the test numbers stay fair. Add
`--trials-dir data\hashing_trials_e512` to score a run saved somewhere else.
Without `--top-trials`, the script scores only `data\hashing_model.pt`.

### Results so far

These are test results with 256-bit codes on 1,973 test images:

| Method | precision@10 | Average precision |
|---|---|---|
| Random projection | 0.9316 | 0.8712 |
| ITQ | 0.9517 | 0.9399 |
| Learned model (best of 20 short trials, 47 epochs) | 0.9278 | 0.9088 |
| Learned model (one trial, 512 epochs) | 0.9472 | 0.9454 |
| Float embeddings (no hashing) | 0.9540 | 0.9440 |

Longer training made a big difference for the learned model. ITQ is still
slightly better at precision@10, while the learned model is better at average
precision.

One thing to keep in mind: the split divides each person's photos between
training, validation, and test, so the test set contains the same people the
model was trained on. That's why the learned model can score a little above
the float embeddings on average precision. A test on people the model has
never seen would need a split by identity, which the project doesn't do yet.

## Load or resize an image

`image_input.py` loads a regular image and can save a resized copy:

```powershell
python .\image_input.py .\my-image.jpg --resize 224 224 --output .\resized.jpg
```

Without an image path, it loads the first CASIA-WebFace record instead.

## What isn't built yet

- `binary_hashing.py` only searches with random-projection codes. The learned
  codes are saved and scored by `compare_hashing.py`, but no command searches
  with them for a single query yet.
- Searches take a CASIA record key as the query. You can't search with your
  own photo yet.
- There's no live webcam search, no way to register people, and no "unknown"
  result when a match isn't reliable.
