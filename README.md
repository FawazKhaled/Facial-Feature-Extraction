# Facial Feature Extraction

# Possible Methods
- OpenCV’s Haar Cascade classifier
- Dlib’s HOG + SVM frontal face detector
- Dlib’s CNN face detector, and Mediapipe’s face detector

# 2 Ways to Extract Features
- hand-crafted features three global features were extracted: Scale-Invariant Feature Transform (SIFT), Speeded Robust Features (SURF), and Global Image Structure (GIST). Likewise, the following local feature methods are utilized: Local Binary Pattern (LBP), Weber local descriptor (WLD), and Histogram of Oriented Gradients (HOG)

- deep learning features: convolutional neural networks (CNNs), including VGG16, VGG19, and VGG-Face, and Siamese neural networks (SNNs), which generate face embeddings, ArcFace-Based Deep Hashing.

# Best Proposed Method
- VGG16 (32-bit floating-point-numbers)
- ArcFace-Based Deep Hashing (bits)

# Face comparison prototype

Search CASIA-WebFace using an indexed record as the query:

```powershell
python .\face_comparison.py --query-key 1 --top-k 10
```

The script reads the aligned face images and identity labels from
`data\casia-webface\train.idx` and `train.rec`, excludes the query record, then
ranks other faces by cosine similarity between normalized 512-dimensional
ArcFace embeddings. The first run may take a long time because it extracts the
embeddings; they are cached in the ignored
`data\arcface_embeddings.sqlite3` database for subsequent searches.

Use `--limit 1000` for a quick test that scans at most 1,000 non-query records.
Use `--dataset` to point at a different CASIA-WebFace directory and `--cache`
to choose a different embedding cache path. This prototype compares
floating-point embeddings; it does not yet learn or compare binary hash codes.

## Preliminary result

For query record 1, a search of the first 1,000 non-query records returned ten
top matches, all with the same CASIA identity label as the query (`0000045`).
Cosine similarities ranged from 0.4774 to 0.7313. This is an encouraging
sanity check, not a complete accuracy evaluation: it uses one query and a
limited, identity-ordered portion of the dataset.

## Next steps

- Evaluate queries from multiple identities using precision@10 and mAP.
- Implement binary deep hashing and compare its retrieval results with the
  floating-point ArcFace baseline.

To preview the first indexed CASIA-WebFace image without entering an image
path, run:

```powershell
python .\image_input.py
```

Pass an image path to load a specific image instead:

```powershell
python .\image_input.py "C:\path\to\image.jpg"
```