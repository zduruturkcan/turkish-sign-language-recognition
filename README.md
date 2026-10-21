# Turkish Sign Language (TSL) Recognition — AUTSL Subset

A small isolated sign-language recognition pipeline built on a 20-word subset of the [AUTSL](https://cvml.ankara.edu.tr/datasets/) (Ankara University Turkish Sign Language) dataset. Body and hand landmarks are extracted from each video with MediaPipe, normalized to the signer's body, and fed into three LSTM-based classifiers of increasing complexity: a baseline LSTM, an Attention-LSTM, and a Spatial-Attention-LSTM. The models are compared against each other and against a simple ensemble.

> **Versions.** This README describes the current version (`SIGN_IT.ipynb`). The first version, with the results reported in `SIGN_IT-Paper.pdf`, is preserved in the git tag [`v1-paper`](../../tree/v1-paper). Its results differ substantially; see [Version history](#version-history).

## Pipeline

1. **Data.** 20 Turkish words are sampled from AUTSL (60 videos per word), downloaded via Kaggle. 1,200 clips in total.
2. **Landmark extraction.** Each video frame is run through MediaPipe's Pose Landmarker and Hand Landmarker to produce a 225-dimensional vector per frame (33 pose points + 21 left-hand points + 21 right-hand points, each x/y/z). Every video is resampled to a fixed 30 frames.
3. **Normalization.** Raw MediaPipe coordinates are fractions of the image, so they depend on where the signer stands and how far they are from the camera. Each sequence is re-expressed relative to the signer's own body: the shoulder midpoint becomes the origin and coordinates are divided by shoulder width (median over the frames of the sequence). Undetected landmarks stay exactly zero. This can be switched off with `NORMALIZE_LANDMARKS = False`.
4. **Motion features.** Frame-to-frame velocity is concatenated onto the normalized coordinates, doubling the feature dimension to 450.
5. **Signer-independent split.** Train/validation/test are split by *signer*, not randomly (7 / 2 / 2 signers; 657 / 297 / 246 clips). The model is evaluated on people it never saw during training, which is a harder and more honest test than a random split.
6. **Models.**
   - **Baseline LSTM**: classifies from the final hidden state only.
   - **Attention LSTM**: learns a weight for every frame ("how important was this frame for this sign?") and classifies from the weighted sum.
   - **Spatial-Attention LSTM**: adds a second attention mechanism that weighs *which body part* (pose / left hand / right hand) matters in each frame, before feeding into the same temporal-attention LSTM.
   - **Ensemble**: averages the softmax outputs of the baseline and attention models.
7. **Training.** Adam, 100 epochs, batch size 8, hidden size 64, label smoothing 0.1, weight decay 1e-4. The weights with the best *validation* accuracy are restored before evaluating on the test set.

## Results

Signer-independent split, 246 test clips, 20 classes. Single-run values come from the main run; the 3-seed values are mean ± standard deviation over seeds 42, 123 and 7.

| Method | Accuracy | F1 (macro) |
| --- | --- | --- |
| Baseline LSTM (1 run) | 75.61% | 73.02% |
| Baseline LSTM (3-seed mean ± std) | 79.54% ± 2.76 | 77.04% ± 3.45 |
| Attention LSTM (1 run) | 86.18% | 85.06% |
| Attention LSTM (3-seed mean ± std) | 84.96% ± 2.02 | 83.71% ± 1.79 |
| Spatial+Temporal Attn. (1 run) | 84.15% | 83.90% |
| Spatial+Temporal Attn. (3-seed mean ± std) | 85.09% ± 1.07 | 84.23% ± 0.82 |
| Ensemble (Baseline + Attention, 1 run) | 87.40% | 85.70% |

**What the results show.** Temporal attention improves over the baseline, by about 5 points in the 3-seed mean (85.0% vs 79.5%). The Spatial+Temporal model performs on par with the Attention LSTM (85.1% vs 85.0%) and varies least across seeds. The ensemble is slightly ahead of either model alone in a single evaluation (no seed averaging). With only 3 seeds and two test signers, differences of about 2 points or less (for example Attention vs Spatial) are within the noise.

The hardest words are `vergi`, `hakli` and `nasil`; for the baseline also `persembe` and `yemek`. Landmark detection was not the cause: no class has undetected frames.

## Version history

| | v1 (tag `v1-paper`, `SIGN_IT-Paper.pdf`) | Current |
| --- | --- | --- |
| Landmarks | Raw image coordinates | Shoulder-centered, scaled by shoulder width |
| Regularization | Label smoothing and weight decay added after the v1 results were collected | Used for all models |
| Baseline accuracy (1 run / 3-seed) | 53.25% / 51.63% | 75.61% / 79.54% |
| Attention accuracy (1 run / 3-seed) | 60.57% / 49.86% | 86.18% / 84.96% |
| Spatial+Temporal accuracy (1 run / 3-seed) | 33.74% / 37.67% | 84.15% / 85.09% |
| Ensemble accuracy (1 run) | 55.69% | 87.40% |
| Main finding | Attention advantage disappears over 3 seeds; Spatial+Temporal is clearly worst | Attention beats baseline over 3 seeds; Spatial+Temporal is on par with Attention |
| Live webcam demo | In progress | Working (qualitative only) |

Several changes were made between v1 and the current version at the same time (normalization, regularization, checkpoint selection), so these results alone do not show which one is responsible for the improvement. Normalization is the most plausible main contributor, but this has not been tested separately yet.

## A note on evaluation quality

- With only 11 distinct signers and about 33 training clips per class (7 training signers), this is a small dataset for a 20-way sequence classification problem. Single-run results differ from the 3-seed means (for example the baseline scores 75.6% in its main run and 79.5% on average), so the multi-seed numbers are the more reliable comparison.
- The test set has only two signers, so results depend on who they are.
- The confusion matrices and classification reports pin `labels=range(num_classes)` instead of letting scikit-learn infer them from the classes present in a given run, so the matrix cannot silently shrink and shift the axis labels.
- The label smoothing and weight decay settings are there to reduce overconfidence on this small dataset.
- A per-class check for landmark-detection failures is included after the class-balance cell, so a badly performing class can be checked against data quality before blaming the model.

## Live webcam demo

The notebook ends with an optional live demo: record yourself signing in Colab and get a prediction from the Attention LSTM. A framing check (guide box, distance and centering compared with the training videos) runs before the 3-second recording, and landmarks are extracted in the same MediaPipe `VIDEO` mode as in training. The demo is qualitative: accuracy on live footage is expected to be lower than on the AUTSL test signers, because a personal webcam differs from the dataset's studio setup in camera, lighting, framing and background.

## Setup

This notebook is written for **Google Colab** (it uses `google.colab.drive`; it has not been adapted for local execution).

1. Open the notebook in Colab.
2. Run the setup cell, then **restart the runtime** if prompted (MediaPipe needs a fresh runtime after installation).
3. Mount Google Drive when prompted. The dataset, extracted landmark arrays and trained models are cached there across sessions.
4. You need a Kaggle account and an API token (`kaggle.json`) to download the [AUTSL dataset from Kaggle](https://www.kaggle.com/datasets/sttaseen/autsl). The notebook asks you to upload it the first time and reuses it afterwards. Never commit `kaggle.json` to a repository.
5. Run the notebook top to bottom. Landmark extraction over about 1,200 videos takes roughly 1.5–2 hours; afterwards `X.npy`, `y.npy` and related files are saved to Drive, and later sessions can skip to the **"Resume from saved arrays"** cell.

If you reopen the notebook in a new session, restart the runtime and run it top to bottom rather than jumping into the middle. The notebook mixes "run fresh" cells with "resume from Drive" cells, and running them out of order can cause stale variables or dimension mismatches.

## Future work

- Ablation: rerun the 3-seed comparison with `NORMALIZE_LANDMARKS = False` (everything else unchanged) to measure how much of the improvement over v1 comes from normalization.
- Run more seeds to narrow the confidence intervals, and investigate why `vergi`, `hakli` and `nasil` remain the weakest classes.
- Add a few own webcam recordings per word to training to reduce the studio-to-webcam gap.
- Explore a graph-convolutional (ST-GCN-style) treatment of the landmark skeleton as an alternative to the flattened per-frame vector.

## Acknowledgements

- [AUTSL dataset](https://cvml.ankara.edu.tr/datasets/): Ankara University Turkish Sign Language dataset. Please review AUTSL's own license/terms before any use beyond this kind of educational/research project, and cite the original dataset if you use it in your own work.
- [MediaPipe](https://developers.google.com/mediapipe): Pose Landmarker and Hand Landmarker models used for feature extraction.
