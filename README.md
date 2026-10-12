# Face search with ArcFace and hashing

I’m working on a face search project. The idea is to turn each face into a
list of numbers, then make that list shorter so faces can be compared with
compact codes.

## What works so far

The project can read the CASIA-WebFace dataset and compare faces using
ArcFace. For a first test, I searched for record 1 in the first 1,000 records.
The ten closest results had the same identity label as the query.

I also made a basic 256-bit version that compares faces using Hamming
distance. In the current cached sample of 988 faces, its precision@10 was
1.00. These are early results from one query and a small, ordered part of the
dataset, so they do not show how well the method works overall.

## Try it

Run the face comparison:

```powershell
python .\face_comparison.py --query-key 1 --top-k 10 --limit 1000
```

Try the 256-bit baseline on the embeddings already saved:

```powershell
python .\binary_hashing.py --query-key 1 --bits 256 --top-k 10
```

The first command can take a while because it needs to process the images.
It saves finished embeddings so they can be reused. The dataset and cache are
kept out of Git.

## What I plan to do next

The 256-bit codes are currently made with a fixed random projection. The next
step is to train the hashing part so it learns to keep images of the same
person close together. After that, I’ll test it with more identities and
compare it with the current results.
