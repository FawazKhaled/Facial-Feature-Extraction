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

## Results

| What was measured | Result |
|---|---|
| Search quality with ITQ codes (precision@10 / average precision) | 0.9517 / 0.9399 |
| Search quality with the full float embeddings, no hashing | 0.9540 / 0.9440 |
| Size of one face | 32 bytes as a code, 2,048 bytes as a float embedding |
| Strangers accepted at a distance of 90 (CASIA people ITQ never saw) | about 2 in 100,000 pairs |
| A registered face on the laptop webcam | usually 60 to 70 bits away, under the limit of 90 |
| Duplicate registration check (36 CASIA people) | 36 of 36 repeats caught, 0 of 1,260 different pairs flagged |
| Live Face ID speed on the Intel Arc GPU (DirectML) | about 52 ms per frame, roughly 19 updates a second |
| Live Face ID speed on the CPU (3 threads) | about 180 to 270 ms per frame |
| Live Face ID speed on the CPU before limiting threads | about 1,500 ms per frame |

Search quality was measured on 1,973 held-out CASIA test images with 256-bit
codes. ITQ ends up within 0.005 of the uncompressed embeddings on both scores
while using 64 times less space. Step 4 has the full comparison of every
hashing method, including the learned model.

## Project layout

```text
Research/
  face_comparison.py    Search with ArcFace embeddings and fill the embedding cache
  binary_hashing.py     Make random-projection codes and search with them
  hashing_cnn.py        Train the learned hashing model
  compare_hashing.py    Score every method on the same held-out test set
  enroll_face.py        Set up Face ID for a person with the webcam
  recognize_faces.py    Recognize enrolled people live from the webcam
  image_input.py        Load or resize a single image
  dataset_loader.py     List the files in a folder
  lib/
    ArcFace.py          Loads the ArcFace model and extracts embeddings
    caching.py          SQLite storage for embeddings and binary codes
    dataset_reader.py   Reads images from the CASIA-WebFace RecordIO files
    hashing.py          Random projection, ITQ, and Hamming distance helpers
    face_id.py          Webcam, background face processing, and Face ID storage
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

On Windows you can run the face models on the GPU instead, which makes live
Face ID several times faster. Swap the regular ONNX Runtime for the DirectML
build:

```powershell
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-directml==1.24.4
```

The scripts use the GPU automatically when DirectML is installed and fall back
to the CPU otherwise. The codes come out the same either way. `pip check` will
say InsightFace needs `onnxruntime`, but that's only a package name mismatch.
The DirectML build provides the same `onnxruntime` module.

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

## Live Face ID with a webcam

This works like setting up Face ID on a phone. First you register a person,
then the live view puts their name on their face whenever they're in front of
the camera. Faces it doesn't know are labeled "Unknown".

Only bits are ever saved. The webcam frames and the float ArcFace embeddings
stay in memory and are thrown away right after each frame is turned into a
256-bit ITQ code. Face ID uses ITQ instead of the learned model because ITQ
never learned any particular person. The learned model mostly memorized the 58
people it was trained on, so it isn't a safe choice for brand new faces.

### Register a person

```powershell
python .\enroll_face.py --name "Fawaz"
```

A window opens and walks you through five steps: look straight ahead, turn
left, turn right, tilt up, and tilt down. It captures 3 codes per step (change
this with `--samples-per-pose`) and only takes a picture when exactly one face
is in view, close enough, and in the right pose. The bar at the bottom fills
up as it goes. Press Q or Esc to cancel without saving anything.

The codes are saved in `data\face_id.sqlite3`. Registering the same name
again replaces that person's old codes but keeps their original place in line.
The first time you run it, the script also fits ITQ on the cached CASIA
embeddings and saves the settings in `data\itq_256.npz`, which takes about 20
seconds.

The same face can't be registered under two names. After every registration,
and each time the live view starts, the script goes through everyone in the
order they registered. If someone's codes mostly match a person who registered
before them (at least half within a distance of 90), the later one is removed
and the script tells you who it matched. In a test on 36 CASIA people, this
caught every repeat registration and never confused two different people.

### Run live recognition

```powershell
python .\recognize_faces.py
```

Each face gets a box with a name and its Hamming distance to the closest
registered code. A distance of 90 or less counts as a match. Use
`--max-distance` to make it stricter (lower) or more forgiving (higher). In a
test on CASIA people that ITQ never saw, a limit of 90 let through about 2
stranger pairs in every 100,000 and accepted 65% of single same-person pairs.
Registering 15 codes per person makes it much more likely that at least one is
close. Press Q or Esc to quit. Use `--camera 1` if you have more than one
camera.

Face processing takes about 50 ms per frame on the GPU with DirectML, or about
a quarter of a second on the CPU. It runs in the background so the preview
keeps playing at full speed either way. The
boxes can lag a little behind fast movement. Each name only changes once most
of that face's last 5 results agree, so a single blurry frame can't flip it.

If `data\itq_256.npz` is deleted or replaced, the saved codes no longer line
up with new ones. The live view will tell you to register everyone again.

## Load or resize an image

`image_input.py` loads a regular image and can save a resized copy:

```powershell
python .\image_input.py .\my-image.jpg --resize 224 224 --output .\resized.jpg
```

Without an image path, it loads the first CASIA-WebFace record instead.

## License

The code in this repository is released under the [MIT License](LICENSE).

That license only covers this project's code. The CASIA-WebFace dataset and
InsightFace's pretrained ArcFace models (`buffalo_l`) come with their own terms,
which generally allow non-commercial research use only. Neither is included in
this repository.
