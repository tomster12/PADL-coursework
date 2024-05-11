---
tags:
  - Notes
---
# PADL Assessment Notes

---

## Overview

Submission in a zip `Yxxxx.zip`:

- `padl.ipynb` with all code and explanations
- `padl.pdf` as exported notebook
- Exported weights from Q5\*
- Script `predict_time.py` to run Q5\* network

https://drive.google.com/drive/u/1/folders/1LaBLf2ECVTQUS4VS5lNnAaKbhF1pHPoB

---

## 1. PCA (9 marks)

**a. (3 marks)**
Apply [[Principal Components Analysis]] with 5 components to data.
Get minimum number of dimensions needed and explain why.

**b. (6 marks)**
Repeat using the minimum.
Show equations used to find each PCA from original variables.

---

## 2. Regression  (27 marks)

Data in `PADL-Q2-train.csv` header `(x,y,z,w,out)` + 80 rows.
Use sci-kit-learn and $R^2$ value, aiming to train regression.

**a. (9 marks)**
[[Basis Function]], data scaling / normalization.

**b. (9 marks)**
Series of [[Linear Regression]] models, regularisation, [[Piecewise Regression]].

**c. (9 marks)**
Train all above models automatically, select expected best on unseen data.
Paragraph to train best against `PADL-Q2-unseen.csv` and compute $R^2$.

---

## 3. Embeddings  (14 marks)

Text from here https://www.gutenberg.org/cache/epub/48320/pg48320.txt.
Generate slang for words: `L = ["gold","diamond","robbery","bank","police"]`

**a. (3 marks)**
Procedure to extract all triplets of words where middle word is "and".

**b. (3 marks)**
Procedure takes word and gets all triplets with 3rd word with 3+ matching suffix.

**c. (8 marks)**
For each word and associated triplets, word2vec on first and third word in the triplet for semantic similarity. Sort triplets in decreasing similarity and return top 5.
Can use pretrained word2vec model.

---

## 4. Basic MLP (10 marks)

2 inputs 1 output, approximates $x\cdot y$

**a. (4 marks)**
Only linear + ReLU, implement in PyTorch.

**b. (3 marks)**
Appropriate training loop on random data, I choose of loss function.
Explicit comments on assumptions of training data range.

**c. (3 marks)**

Evaluate in terms of absolute error $|f(x,y)-(x\cdot y)|$.
Report mean error over random samples in training data.
Generalisation error on inputs outside of training data.

---

## 5. Telling the time (20 marks)

1000 training data `0000.png - 9999.png` of clocks.
1000 training labels `0000.txt - 9999.txt` in format `HH:MM`.
Labels with $H\in {0..11}$, $M\in {0..59}$.

Task is to predict time from images.

- Performance = absolute difference in minutes
- Train performance = median absolute difference in minutes
- Input Bx3x448x448 scaled to the range (0,1)
- Output Bx2 in ranges (0..11, 0..59) for (hour, minute)
- Exported weights < 20MiB

Need to export weights to a file and make a `predict_time.py`.

- `predict_time.py` needs a function `predict(images)`.
- `predict` should load weights, pass input through network, then return output.

Include training / validation code, as well as design discussion.

**a. (5 marks)**
Data Loader for clocks dataset.
Can pre-process or augment, but must explain.

**b. (4 marks)**
Design and implement appropriate architecture.

**c. (3 marks)**
Design and justify appropriate loss function.

**d. (3 marks)**
Plot training and validation losses and use this to justify hyperparameter choices.

**e. (5 marks)**
Performance on unseen test set, median error < 5 minutes = 5 marks.

---

## 6. Generative (20 marks)

Same dataset as Q5 only images, creating realistic clock images.
E.g. Generative Adversarial Network, Variational Autoencoder, Diffusion Model.

Paragraph to generate 8 images (sample from latent space, process, output).

Paragraph to perform latent space interpolation between 2 samples.

- Generate 2 samples
- Linearly interpolate 5 intermediate latent vectors
- Display the 7 images in order.

**a. (5 marks)**
Design and implement appropriate architecture with justification.

**b. (5 marks)**
Appropriate pre-processing, training loop implementation, hyperparameters.

**c. (5 marks)**
Generate 8 clock images successfully.

**d. (5 marks)**
Interpolate successfully.
