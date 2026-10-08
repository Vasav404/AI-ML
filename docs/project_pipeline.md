# Image Classification Pipeline (Planned — Not Yet Implemented)

This is the **expected future pipeline** for the project. Nothing here has been executed yet;
it defines how we will go from raw dataset → trained model → predicted medicinal plant.

## The pipeline at a glance

```
Dataset
   |
   v
Data Cleaning
   |
   v
Train / Validation / Test Split
   |
   v
Image Resizing
   |
   v
Normalization
   |
   v
Data Augmentation
   |
   v
Feature Extraction / CNN
   |
   v
Model Training
   |
   v
Validation
   |
   v
Testing
   |
   v
Evaluation
   |
   v
Predicted Medicinal Plant
```

## Step-by-step explanation

### 1. Dataset
The raw images in `data/raw/` (80 class folders, 6,900 images — see
[dataset_research.md](dataset_research.md)). Read-only; never edited in place.

### 2. Data Cleaning
- Remove exact duplicates (found by `scripts/explore_dataset.py` via SHA-256).
- Remove/repair corrupt or unreadable files (Pillow `verify()` failures).
- Fix obvious label issues (folder-name typos are cosmetic but documented).
- Optionally remove images with wrong species (spot-check a sample visually).
- Optionally segment the leaf from the background (a *possible* upgrade — not required).

### 3. Train / Validation / Test Split
- **Stratified** split so every class appears in every part (critical with only 8 images in
  some classes) — suggested ratio **70% train / 15% validation / 15% test**.
- Split **before** augmentation (never augment first — causes leakage).
- Keep near-duplicate photos of the same leaf in the **same** split.
- Test set is touched **once**, at the very end.

### 4. Image Resizing
- All images resized to a fixed size, typically **224×224** (matches MobileNet/ResNet/EfficientNet
  input expectations; VGG too). Aspect ratio: either stretch or center-crop — decide and keep
  consistent; document the choice.
- Why: CNNs need fixed-size input batches; also neutralizes the dataset's mixed resolutions
  (450–4656 px).

### 5. Normalization
- Scale pixel values from `[0, 255]` → `[0, 1]` (or standardize).
- For transfer learning use **ImageNet mean/std** (mean `[0.485, 0.456, 0.406]`, std
  `[0.229, 0.224, 0.225]`) because the pretrained weights expect that distribution.
- Outputs go to `data/processed/` (or are produced on-the-fly by the data loader).

### 6. Data Augmentation
Only on the **training** split: random horizontal/vertical flips, rotation (±30°), zoom,
brightness/contrast jitter, slight shear, cutout/random erasing.
- Purpose: teaches rotation/lighting invariance (challenges #2–#4 in the README) and
  combats class imbalance — small classes get more variety each epoch.
- Not applied to validation/test (they must stay realistic).

### 7. Feature Extraction / CNN
- **Classical route:** hand-made features (HOG, LBP, color histograms, GLCM) per image.
- **Deep route:** a CNN learns features automatically; with transfer learning the pretrained
  backbone extracts features and a new 80-class head classifies them.
- Details and model comparison: [model_research.md](model_research.md).

### 8. Model Training
- Loss: **categorical cross-entropy** (use `class weights` or balanced sampling for imbalance).
- Optimizer: Adam (lr ≈ 1e-3 for a new head, ≈1e-4 for fine-tuning) or SGD+momentum.
- Phases (transfer learning): (a) train head with backbone frozen → (b) unfreeze top blocks and
  fine-tune with a small learning rate.
- Batch size 16–64 depending on hardware; a few epochs per phase is usually enough.
- Everything (framework, seed, hyperparameters) logged for **reproducibility**.

### 9. Validation
- Evaluate on the validation split **every epoch**.
- Track validation loss/accuracy to detect overfitting (train ↑ but val ↓).
- **Early stopping** (e.g. patience = 5–10 epochs) + checkpoint the best model.
- Use it for model selection and hyperparameter tuning — never for final claims.

### 10. Testing
- Run the **best checkpoint once** on the untouched test split.
- Produces the honest, final estimate of real-world performance.

### 11. Evaluation
- **Accuracy** (top-1), **macro-F1** (main metric — classes are imbalanced 8…177),
  weighted-F1, per-class precision/recall/support.
- **Confusion matrix** (80×80 heatmap) → reveals look-alike species pairs.
- Compare models fairly (same split, same preprocessing): SVM vs RF vs KNN vs Custom CNN vs
  MobileNet vs ResNet vs EfficientNet.
- Optional: Grad-CAM visualizations to explain predictions.

### 12. Predicted Medicinal Plant
Final user-facing flow:

```
new leaf photo -> resize 224x224 -> normalize -> model.predict
            -> softmax over 80 classes -> argmax
            -> "Neem" (confidence 0.94)
```

A tiny inference script (image path → label) may be added **only when instructed**;
no web app / UI in the current scope.

## Where each step will live

| Step | Planned location |
|---|---|
| raw data (never modified) | `data/raw/` |
| cleaning + split + resize/normalize outputs | `data/processed/` (future scripts) |
| augmentation (on-the-fly) | training data loader (future) |
| training/evaluation code | `scripts/` or `notebooks/` (future, when allowed) |
| metrics, confusion matrix, plots | `reports/` (future) |

## Rules that stay in force

1. Raw data is read-only — cleaning writes to `data/processed/`.
2. No leakage: split before augmentation; duplicates stay together.
3. Test set is evaluated once, at the end.
4. Every run records: dataset version, split seed, model, hyperparameters.
5. **Training starts only when explicitly instructed** — current phase stops at research.

