# Turkish Sign Language (TSL) Recognition — AUTSL Subset

A small isolated sign-language recognition pipeline built on a 20-word subset of the
[AUTSL](https://cvml.ankara.edu.tr/datasets/) (Ankara University Turkish Sign
Language) dataset. Body and hand landmarks are extracted from each video with
MediaPipe, then fed into three LSTM-based classifiers of increasing complexity —
a baseline LSTM, an Attention-LSTM, and a Spatial-Attention-LSTM — which are
compared against each other and a simple ensemble.

## Pipeline

1. **Data.** 20 Turkish words are sampled from AUTSL (60 videos per word where
   available), downloaded via Kaggle.
2. **Landmark extraction.** Each video frame is run through MediaPipe's Pose
   Landmarker and Hand Landmarker to produce a 225-dimensional feature vector per
   frame (33 pose points + 21 left-hand points + 21 right-hand points, each x/y/z).
   Every video is resampled to a fixed 30 frames.
3. **Motion features.** Frame-to-frame velocity is concatenated onto the raw
   coordinates, doubling the feature dimension to 450.
4. **Signer-independent split.** Train/validation/test are split by *signer*, not
   randomly — the model is evaluated on people it never saw during training, which
   is a meaningfully harder (and more honest) test than a random split.
5. **Models.**
   - **Baseline LSTM** — classifies from the final hidden state only.
   - **Attention LSTM** — learns a weight for every frame ("how important was this
     frame for this sign?") and classifies from the weighted sum.
   - **Spatial-Attention LSTM** — adds a second attention mechanism that also
     weighs *which body part* (pose / left hand / right hand) matters per frame,
     before feeding into the same temporal-attention LSTM.
   - **Ensemble** — averages the softmax outputs of the baseline and attention
     models.

## Results

Full write-up, methodology, and confusion-matrix analysis are in
[`SIGN_IT-Paper.pdf`](./SIGN_IT-Paper.pdf). Table 3 from that report, reproduced
here — single-run accuracy/F1 alongside the more reliable 3-seed mean ± std
(signer-independent split, 246 test examples across 20 classes):

| Method                              |     Accuracy     |    F1 (macro)     |
|--------------------------------------|:-----------------:|:------------------:|
| Baseline LSTM (1 run)                |       53.25%       |       48.16%       |
| Baseline LSTM (3-seed mean ± std)    |   51.63% ± 3.91    |   47.65% ± 3.80    |
| Attention LSTM (1 run)               |       60.57%       |       59.37%       |
| Attention LSTM (3-seed mean ± std)   |   49.86% ± 1.64    |   44.99% ± 2.55    |
| Spatial+Temporal Attn. (1 run)       |       33.74%       |       30.13%       |
| Spatial+Temporal Attn. (3-seed mean ± std) | 37.67% ± 4.46 |   34.26% ± 4.32    |
| Ensemble (Baseline + Attention, 1 run) |     55.69%        |       51.19%       |

**The headline finding is in the 3-seed numbers, not the single-run ones.** A single
training run makes the Attention-LSTM look like the clear winner (+7.3 accuracy
points over the baseline). Averaged over three random seeds, that advantage
disappears — the Baseline's 3-seed mean (51.6% ± 3.9) is actually marginally
*higher* than the Attention-LSTM's (49.9% ± 1.6), and their standard-deviation
ranges overlap almost entirely. What *does* hold up under multi-seed evaluation:
the Attention-LSTM is consistently more stable (roughly half the baseline's
standard deviation), and the Spatial+Temporal model is the one architecture that's
robustly worse than both others under every protocol tested (single-run, 3-seed,
and ensemble) — its confusion matrix shows a mode-collapse pattern, with many
unrelated classes misclassified into a small handful of "default" predictions,
most visibly `turkiye`.

Two words — `hafif` ("light") and `hali` ("carpet") — were consistently the
hardest to classify across every model and every seed.

## A note on evaluation quality

A few things are worth knowing about this setup before over-trusting any single
number above:

- With only ~11 distinct signers and ~30 training examples per class, this is a
  genuinely small dataset for a 20-way sequence classification problem — the
  paper's own central methodological finding is that single-run comparisons
  between closely-matched architectures are unreliable at this scale and can
  reverse under multi-seed re-evaluation (see above).
- The confusion matrix and classification report explicitly pin `labels=range(num_classes)`
  rather than letting scikit-learn infer them from whichever classes happen to
  appear in a given run. With a non-stratified, signer-independent split, a class
  can end up with very few (or zero) test examples in a given run; without pinning
  the labels, the matrix can silently shrink and shift every later class's axis
  label by one.
- Training uses `label_smoothing=0.1` and `weight_decay=1e-4` specifically to fight
  the overconfidence you'd otherwise get from cross-entropy on this little data —
  i.e. the model outputting a very high probability for the wrong class, which is a
  calibration failure, not just an accuracy one. (This was added after the results
  above were collected, so re-running the notebook now may shift these numbers
  somewhat — the qualitative story, especially the single-run-vs-3-seed gap, is
  expected to hold.)
- One class (`misafir` in early runs) scored 0.00 precision/recall across every
  model. A per-class landmark-detection-failure check is included right after the
  class-balance cell — worth running before assuming a badly-performing class is a
  modeling problem rather than a data-quality one.

## Setup

This notebook is written for **Google Colab** (it uses `google.colab.drive` and
expects a GPU/CPU runtime there — it hasn't been adapted for local execution).

1. Open the notebook in Colab.
2. Run the setup cell, then **restart the runtime** when prompted (MediaPipe needs
   a fresh runtime after install).
3. Mount Google Drive when prompted — this is where the dataset, extracted
   landmark arrays, and trained models get cached across sessions.
4. You'll need a Kaggle account and an API token (`kaggle.json`) to download the
   [AUTSL dataset from Kaggle](https://www.kaggle.com/datasets/sttaseen/autsl) —
   the notebook will prompt you to upload it the first time and reuse it after
   that.
5. Run the notebook top to bottom. Landmark extraction over ~1200 videos takes
   roughly 1.5–2 hours; once done, `X.npy`/`y.npy`/etc. are saved to Drive, and
   later sessions can skip straight to the **"Resume from saved arrays"** cell to
   avoid re-extracting.

**If you reopen this notebook in a new session:** restart the runtime and run it
top to bottom rather than jumping to cells in the middle. This notebook mixes
"run fresh every time" cells with "resume from Drive" cells, and running things
out of order is the single most common source of confusing errors here (stale
variables, dimension mismatches between a model built earlier and data added
later, etc.) — see the debugging notes near the top of the notebook for specifics.

## Future work

- A live webcam demo (record yourself signing, get a prediction) is in progress —
  early testing showed accuracy on live footage is much lower than on the AUTSL
  test set, which is expected given how different a personal webcam is from the
  dataset's studio recording setup (camera, lighting, framing, and only this one
  dataset's signers to generalize from). A pre-recording "framing check" (checking
  distance from camera and centering against the training data's own distribution,
  similar to how a document-scanner app guides you to align a page) is being
  added to reduce that gap before this ships.
- Run additional seeds to narrow the confidence intervals in the results table
  further, and investigate directly why `hafif`/`hali` and the Spatial+Temporal
  model's mode-collapse classes remain unrecognized.
- Explore a graph-convolutional (ST-GCN-style) treatment of the landmark skeleton
  as a stronger alternative to the flattened per-frame vector used throughout this
  project.

## Acknowledgements

- [AUTSL dataset](https://cvml.ankara.edu.tr/datasets/) — Ankara University
  Turkish Sign Language dataset. Please review AUTSL's own license/terms before
  any use beyond this kind of educational/research project, and cite the original
  dataset if you use it in your own work.
- [MediaPipe](https://developers.google.com/mediapipe) — Pose Landmarker and Hand
  Landmarker models used for feature extraction.
