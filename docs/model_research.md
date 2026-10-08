# Model Research — Possible ML/DL Approaches for Leaf Classification

**Status: RESEARCH ONLY — nothing in this document has been trained.**
This document compares models we *could* use in the next phase for
**80-class medicinal leaf image classification** (6,900 images, see [dataset_research.md](dataset_research.md)).

---

## 0. Problem framing for model choice

| Property | Value | Consequence for models |
|---|---|---|
| Task | single-label, **80-class** image classification | need model + loss for multi-class (softmax / one-vs-rest) |
| Data size | **6,900** images (~86/class avg, min 8) | moderate-small → deep nets from scratch risk overfitting; classical ML viable with good features |
| Class balance | **imbalanced** (8–177 per class) | use class weights, macro-F1, stratified splits |
| Image sizes | mixed (450 px … 4656 px) | must resize (e.g. 224×224) before modeling |
| Compute available | student laptop / free Colab (assumption) | prefer MobileNet/EfficientNet-B0; keep VGG as optional |

Two families of approaches:

```
A) Classical ML:   image -> hand-crafted features (HOG, LBP, color, shape) -> SVM / RF / KNN
B) Deep learning:  image -> learned features (CNN) ----------------------> softmax classifier
```

---

## 1. Traditional Machine Learning approaches

### 1.1 Feature extractors used with classical ML
*(needed because SVM/RF/KNN do not learn features by themselves)*

- **HOG** (Histogram of Oriented Gradients) — edge/shape directions; captures leaf outline & margin.
- **LBP** (Local Binary Patterns) — micro-texture (hairy, smooth, waxy surfaces).
- **GLCM** (gray-level co-occurrence) — contrast, homogeneity, roughness.
- **Color histograms** — green/yellow/purple tint distributions.
- **Shape descriptors** — area, perimeter, circularity, elongation, Hu moments (needs segmentation).
- **pCA / flattening** — pixel vectors as a baseline feature.

### 1.2 Support Vector Machine (SVM)

| | |
|---|---|
| **Basic idea** | Finds the hyperplane that separates classes with the **maximum margin**; kernel trick (RBF, poly) handles non-linear boundaries. Multiclass via one-vs-one. |
| **Use for leaves** | Extract HOG + color + LBP features (optionally PCA-reduced) → train an RBF-SVM over the 80 classes. |
| **Advantages** | Often excellent accuracy on small/medium image-feature sets; robust to overfitting when regularized; well-understood theory. |
| **Disadvantages** | No end-to-end learning (features must be hand-made); training time grows fast with samples (O(n²–n³)); probability calibration is clumsy; per-class decision functions. |
| **Compute** | Trains fine on a laptop CPU for ~7k samples × few hundred features (seconds–minutes). |
| **Suitability** | ⭐⭐⭐⭐ **Strong classical baseline** — great "non-deep-learning" comparison point for the report. |

### 1.3 Random Forest (RF)

| | |
|---|---|
| **Basic idea** | Ensemble of many decision trees trained on random subsets of data/features; final vote = prediction. |
| **Use for leaves** | Same hand-crafted features as SVM; also handy for **feature importance** (which characters matter). |
| **Advantages** | Fast, minimal tuning, resistant to overfitting/noise, handles unscaled features, interpretable importances. |
| **Disadvantages** | Trees rarely beat SVM/DL on complex visual patterns; pixel-level features confuse it; forests with deep trees get big. |
| **Compute** | Very cheap on CPU. |
| **Suitability** | ⭐⭐⭐ Good, quick baseline + interpretability section for the report. |

### 1.4 K-Nearest Neighbors (KNN)

| | |
|---|---|
| **Basic idea** | "You are what your neighbors are": classify by the majority label among the *k* closest training vectors (usually Euclidean distance). |
| **Use for leaves** | Features from §1.1; standardize/PCA first; try k = 1…15. |
| **Advantages** | Zero training cost, dead simple, good sanity check, sometimes surprisingly strong with PCA features. |
| **Disadvantages** | Slow & memory-hungry prediction (stores all samples); sensitive to irrelevant features, scaling and the "curse of dimensionality"; struggles with 80 classes in high dimensions. |
| **Compute** | CPU + RAM; fine at 6,900 samples. |
| **Suitability** | ⭐⭐ Quick baseline only — expect it to be weakest of the three. |

---

## 2. Deep Learning approaches

### 2.1 Custom CNN (built from scratch)

| | |
|---|---|
| **Basic idea** | Stack Conv → ReLU → Pool layers so the network **learns edges → textures → leaf patterns** on its own, ending in a 80-way softmax. |
| **Use for leaves** | e.g. 4–5 conv blocks (32→64→128→256 filters) + GlobalAveragePool + Dropout(0.5) + Dense(80); train with categorical cross-entropy. |
| **Advantages** | Teaches the fundamentals; no pretrained weights needed; fast epochs; fully controlled; surprisingly decent when combined with augmentation. |
| **Disadvantages** | Needs careful tuning (depth, dropout, LR); on 6,900 images it can overfit; rarely matches transfer learning on accuracy. |
| **Compute** | Trainable on a laptop CPU (slower) or any free GPU (fast). |
| **Suitability** | ⭐⭐⭐⭐ **Essential teaching baseline** — shows the "learning from scratch" gap vs transfer learning. |

### 2.2 VGG (VGG16)

| | |
|---|---|
| **Basic idea** | Very simple, uniform design: many 3×3 conv layers (Simonyan & Zisserman, 2015); ~138M parameters. |
| **Use for leaves** | Use pretrained VGG16 as a **fixed feature extractor** (images → 4096-d features) → classifier; or fine-tune last blocks. |
| **Advantages** | Extremely simple architecture; strong low/mid-level features; great for teaching and for extracting features for SVM. |
| **Disadvantages** | Huge & slow (≈500 MB weights, fully-connected layers); older accuracy/speed trade-off; heavy on laptops. |
| **Compute** | GPU recommended; CPU inference OK, training slow. |
| **Suitability** | ⭐⭐ Optional — feature-extraction experiments only; not the main model. |

### 2.3 ResNet (ResNet50)

| | |
|---|---|
| **Basic idea** | **Residual (skip) connections** let very deep networks train without degradation (He et al., 2016); ResNet50 ≈ 25.6M params, commonly ~76% top-1 on ImageNet. |
| **Use for leaves** | Load ImageNet weights → replace classifier head with 80 outputs → fine-tune with low learning rate. |
| **Advantages** | Deep = rich features; trains reliably thanks to skip connections; huge ecosystem (tutorials, pretrained weights, tips); strong general accuracy. |
| **Disadvantages** | Bigger/slower than MobileNet/EfficientNet; full fine-tuning needs a GPU; can still overfit small classes without augmentation. |
| **Compute** | Free Colab GPU is comfortable; CPU training feasible but slow. |
| **Suitability** | ⭐⭐⭐⭐ **Reliable main baseline** — best "standard, well-cited" choice. |

### 2.4 MobileNet (MobileNetV2 / V3)

| | |
|---|---|
| **Basic idea** | Lightweight CNNs built from **depthwise-separable convolutions** (+ inverted residuals in V2) — ~3.4M params (V2) instead of tens of millions (Sandler et al., 2018). |
| **Use for leaves** | Same transfer-learning recipe as ResNet; designed to end up **on a phone** — matches the "field identification app" story. |
| **Advantages** | Tiny, fast, CPU-friendly; low memory → deployable later on mobile/edge; accuracy per parameter is excellent. |
| **Disadvantages** | Slightly lower raw accuracy than ResNet/EfficientNet; depthwise convs can under-utilize some GPUs. |
| **Compute** | Runs happily on CPU; training is fast even on free Colab. |
| **Suitability** | ⭐⭐⭐⭐⭐ **Best fit for this project** (student hardware + future mobile demo). |

### 2.5 EfficientNet (B0)

| | |
|---|---|
| **Basic idea** | **Compound scaling** of depth/width/resolution together; EfficientNet-B0 ≈ 5.3M params with ~77% ImageNet top-1 (Tan & Le, 2019) — accuracy-per-FLOP record when published. |
| **Use for leaves** | Transfer learning exactly like ResNet/MobileNet; B0 for laptops, larger B1–B7 if a GPU appears. |
| **Advantages** | Best accuracy-to-size trade-off; modern architecture; fewer params than ResNet50 for better accuracy. |
| **Disadvantages** | Heavier inputs (224×224) and more activation memory than MobileNet; scaling variants grow fast. |
| **Compute** | B0 trains fine on free Colab GPU; CPU workable. |
| **Suitability** | ⭐⭐⭐⭐ **Accuracy-focused candidate** — worth trying after MobileNet/ResNet. |

---

## 3. Why **transfer learning** is especially useful here

Transfer learning = reuse a network already trained on **ImageNet** (1.2M photos, 1,000 classes)
and adapt it to our 80 leaf classes.

1. **Our dataset is modest.** 6,900 images for 80 classes is *not much* for learning a visual
   vocabulary from scratch. ImageNet weights already encode edges → textures → shapes → parts —
   exactly the hierarchy leaf recognition needs.
2. **Much faster convergence.** A MobileNetV2/ResNet50 head can reach a good accuracy in a few
   epochs on a free Colab GPU (or acceptable accuracy on CPU), vs hundreds of epochs from scratch.
3. **Less overfitting.** A mostly-frozen pretrained backbone acts as a strong regularizer —
   important for the tiny classes (8 images for Lemongrass/Pepper!).
4. **Better accuracy, less tuning.** Transfer learning is standard practice in modern plant
   recognition projects; it typically beats small custom CNNs with far less architecture search.
5. **Cost:** one `pip install torchvision` (or `tensorflow.keras.applications`) and free pretrained
   weights — negligible extra compute.

```
ImageNet-pretrained backbone  ->  frozen at first (feature extractor)
        + new Dense(80, softmax) head  ->  train head (fast)
        ->  then unfreeze top blocks and fine-tune with tiny learning rate
```

## 4. Comparison table (all candidates)

| Model | Type | Params (approx.) | Train on laptop CPU? | Free Colab GPU? | Expected role in this project |
|---|---|---|---|---|---|
| **SVM (+HOG/colors)** | classical | n/a (feature-dim) | ✅ easy | n/a | interpretable classical baseline |
| **Random Forest** | classical | n/a | ✅ easy | n/a | baseline + feature importance |
| **KNN** | classical | n/a (stores data) | ✅ easy | n/a | sanity check (weakest) |
| **Custom CNN** | DL from scratch | 1–10 M (design-dependent) | ⚠️ slow but possible | ✅ | "scratch vs pretrained" comparison |
| **VGG16** | DL pretrained | ~138 M | ⚠️ heavy | ✅ | optional feature extractor |
| **ResNet50** | DL pretrained | ~25.6 M | ⚠️ slow | ✅ | robust main baseline |
| **MobileNetV2** | DL pretrained | ~3.4 M | ✅ good | ✅ | **primary model** (speed + deployability) |
| **EfficientNet-B0** | DL pretrained | ~5.3 M | ⚠️ ok | ✅ | accuracy-chasing model |

*Parameter counts are the standard published values for the named architectures (approximate).*

## 5. Suggested experiment plan for the NEXT phase (not executed yet)

**Order of work when training is allowed:**

1. **Classical baselines:** HOG + color-hist (+LBP) → SVM (RBF), Random Forest, KNN.
   *Why:* fast, CPU-only, gives a reference line for the DL models.
2. **Custom CNN** (small, with augmentation + dropout) → shows what scratch learning achieves.
3. **Transfer learning:** MobileNetV2 (primary) → ResNet50 → EfficientNet-B0, same split & preprocessing.
4. **Fair comparison protocol:**
   - identical **stratified** train/val/test split (e.g. 70/15/15),
   - identical resize (224×224) and augmentation policy,
   - early stopping on validation loss.

**Evaluation metrics (because of imbalance):**
- top-1 **accuracy**, **macro-F1** (main metric), weighted-F1,
- per-class precision/recall, **confusion matrix** (find look-alike pairs),
- training time / model size (for the deployability story),
- later: **Grad-CAM** heatmaps to show *what* the model looks at (explainability).

**Anticipated problems & mitigations:** tiny classes → stratified split + augmentation + class
weights (or few-shot handling); background bias → background-heavy augmentation, cutout;
duplicates → removed by `explore_dataset.py` findings before splitting.

---

**Reminder:** per assignment rules, **no model has been trained** — this is a comparison study
only. Framework choice (torchvision vs tensorflow/keras) will be decided when training begins.

---
*Document status: research complete, implementation pending instructor approval.*



