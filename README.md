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