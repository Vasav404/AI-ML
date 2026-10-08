# Medicinal Plant Identification from Leaves

**Image classification project — identify medicinal plant species from leaf images.**

| | |
|---|---|
| **Current phase** | Problem understanding, dataset collection & verification, model research |
| **Model training** | **NOT started yet** (by design — see assignment scope) |
| **Selected dataset** | *Indian Medicinal Leaves Image Datasets* (a.k.a. **DIMPSAR**) — CC BY 4.0; published: 80 classes / 6,900 images |
| **Dataset status** | ✅ **Downloaded & locally verified** (Kaggle mirror): **40 classes / 5,945 JPEGs / 0 corrupt / 57 duplicate groups** → see `reports/dataset_summary.txt` |

---

## 1. What "Medicinal Plant Identification from Leaf Images" Means

Medicinal plants are species used to make herbal medicines, Ayurvedic/ASU formulations,
essential oils and traditional remedies. Many medicinal species look similar to the
untrained eye, and misidentification can be dangerous (some look-alike plants are toxic)
or economically harmful (adulteration in the herbal industry).

**Medicinal plant identification from leaf images** means: given a photograph of a leaf,
a computer program decides **which plant species** the leaf belongs to — automatically,
without an expert botanist looking at it every time.

## 2. Why This Is an Image Classification Problem

**Image classification** = the computer is given **one image** and must choose **one label
from a fixed list of classes**.

- The input is a single leaf image (pixels).
- The output is one class out of the 80 species in our dataset.
- The model learns visual patterns (shape, margin, veins, color, texture) that separate
  one species from another — exactly what a human expert does, but at scale.

It is *not* object detection (we are not drawing boxes around many leaves) and *not*
segmentation (we are not outlining the leaf boundary), although those could be added later.

## 3. Input and Output

```
INPUT                                              OUTPUT
------                                             ------
A leaf photo:                                       One class label out of 80,
- format: JPEG / PNG (RGB)                          e.g. "Neem" or "Tulsi"
- size: varies (e.g. 450x600 px seen in the        + (later) a confidence score,
  verified sample of our dataset)                   e.g. Neem: 0.94
- typically one leaf per image
```

**Classes (labels):** the 80 medicinal-plant leaf folders of the selected dataset —
e.g. Aloevera, Amla, Betel, Bhrami, Curry, Henna, Hibiscus, Mint, Neem, Tulsi,
Turmeric, … (full verified list in [docs/dataset_research.md](docs/dataset_research.md)).

## 4. How Leaf Characteristics Distinguish Plant Species

A trained model (or expert) separates species using leaf "characters":

| Characteristic | What it looks like | Example |
|---|---|---|
| **Leaf shape (outline)** | oval, lance-shaped, heart-shaped, needle-like, lobed | Mango = lanceolate, Betel = heart-shaped |
| **Leaf margin (edge)** | smooth, serrated (saw-toothed), wavy, lobed | Neem has toothed leaflets |
| **Venation (vein pattern)** | parallel vs net-like (reticulate), midrib strength | Monocots (Turmeric, Lemongrass) = parallel veins |
| **Color** | green shades, yellow/green tints, purplish underside | Holy basil (Tulsi) often has purplish tint |
| **Texture & surface** | glossy, hairy, waxy, rough, hairy margin | Mint = textured, slightly hairy |
| **Size & proportion** | length-to-width ratio, leaflet count | Curry = compound, many small leaflets |
| **Leaf arrangement / compound vs simple** | single blade vs divided leaflets | Drumstick = compound, fine leaflets |

Deep learning models learn these features **automatically** from pixels
(this is the key advantage over classical ML, which needs hand-made features).

## 5. Possible Challenges

| # | Challenge | Why it hurts |
|---|---|---|
| 1 | **Similar-looking leaves** | Closely related species (e.g. two mints, two basils) differ only in fine details → confused classes. |
| 2 | **Different lighting conditions** | Sunlight, shade, flash, indoor bulbs change colors and shadows → model may learn lighting instead of species. |
| 3 | **Different backgrounds** | White paper, soil, grass, cluttered tables → model may cheat by learning the background. (Our dataset was captured "with varying background without any environment constraints" — real, but harder.) |
| 4 | **Leaf orientation** | Leaves photographed sideways/upside-down appear different although they are the same class → needs rotation invariance / augmentation. |
| 5 | **Multiple leaves in one image** | Model expects one leaf; several leaves or overlapping leaves break that assumption. |
| 6 | **Different growth stages** | Young leaves are smaller/lighter, old leaves damaged/darker → same species looks different over time. |
| 7 | **Image quality/resolution** | Blur, compression artifacts, low light, very mixed resolutions (our verified sample spans 450 px up to 4656 px). |
| 8 | **Class imbalance** | Some classes have many more images than others (verified: 8–177 images per class in our dataset) → model biased toward big classes; a few classes (Lemongrass: 8, Pepper: 8) are very hard to learn. |

Other real risks: near-duplicate photos of the *same* leaf inflating accuracy, label typos
in folder names, and domain gap between dataset photos and photos a user would take.

## 6. Real-World Applications

- **Herbal & Ayurvedic industry:** raw-leaf intake quality control, spotting adulteration.
- **Pharmacognosy & drug research:** quick verification of plant material before extraction.
- **Field identification apps:** phone camera → species name for foresters, farmers, students.
- **Biodiversity conservation:** rapid documentation of medicinal species (incl. threatened ones).
- **Education:** botany/Pharmacy students learn species recognition interactively.
- **Digitized herbaria:** automating labeling of large leaf-image collections.
- **Traditional medicine systems:** supporting consistent, safe use of herbs (Ayurveda, Siddha, Unani).

## 7. Limitations of Image-Based Plant Identification

1. **It is a visual aid, not an authority.** Leaves alone sometimes cannot separate
   look-alike species; experts also use flowers, fruit, bark, smell and taste.
2. **Same species ≠ same chemistry.** Two plants can be the same species but differ in
   medicinal potency (soil, season, region). Image models identify *species*, not *chemical composition*.
3. **Dataset bias.** A model trained on 80 Indian species with certain backgrounds fails on
   new regions/backgrounds/cameras (out-of-distribution data).
4. **Not medically certified.** Nothing here should be used for medical dosing decisions;
   it is a college ML project, not a diagnostic device.
5. **Errors are not uniform.** With 80 classes, some confusions are rare but serious
   (e.g., medicinal vs toxic look-alike) — confidence thresholds and expert review matter.
6. **Coverage.** The dataset covers a small fraction of the thousands of medicinal plants
   worldwide; the model can only ever say one of its 80 classes.

## 8. How the System Works — Simple Example

```
        Leaf image
             |
             v
   +---------------------+
   | Image preprocessing |   (resize, normalize, denoise, ...)
   +---------------------+
             |
             v
   +-----------------------------------+
   | Feature extraction / Deep Learning|   (HOG+classical ML, or CNN features)
   +-----------------------------------+
             |
             v
   +---------------------+
   |     Classification  |   (one of 80 species)
   +---------------------+
             |
             v
  Predicted medicinal plant / species
```

The step-by-step future implementation pipeline is documented in
[docs/project_pipeline.md](docs/project_pipeline.md).

## 9. Selected Dataset (Quick Facts)

Full comparison of all candidates: **[docs/dataset_research.md](docs/dataset_research.md)**.

| Fact | Value | How it was verified |
|---|---|---|
| Name | *Indian Medicinal Leaves Image Datasets* (paper name: **DIMPSAR**) | Mendeley record + DataCite |
| Classes | **Published card: 80** (folder-per-class labels) · **local Kaggle mirror: 40** | Mendeley/Kaggle description + HF API (`n_unique = 80`) · local inspection script |
| Images | **Published card: 6,900** · **local Kaggle mirror: 5,945** | HF dataset_info (`num_examples = 6900`) · local inspection (0 corrupt) |
| Images per class | card: 8–177 (imbalanced) · local: **115–187 (median 146 — balanced!)** | HF dataset card · local inspection |
| License | **CC BY 4.0** (free for academic use **with attribution**) | DataCite DOI record; Kaggle & HuggingFace metadata |
| Original source | Mendeley Data, V3, DOI **10.17632/748f8jkphb.3** | DataCite |
| Companion paper | Pushpa B R & Rani N S (2023), *Data in Brief* **49**, 109388 | Crossref (DOI 10.1016/j.dib.2023.109388) |
| Format / resolution (local) | **JPEG 100%**; 15 sizes, 277–600 px; common 600×450 & 600×283 | local inspection (Pillow) |
| Duplicates (local) | **57 exact-duplicate groups** → remove before splitting | local SHA-256 scan |
| Local status | ✅ **downloaded & inspected** (route A, no login) → `reports/dataset_summary.txt` | run 2026-09-30 |

> **Mirror caveat (verified, not assumed):** the Kaggle mirror's description says 80
> classes, but the archive actually contains **40 classes / 5,945 images** (locally
> measured). The **full 80-class / 6,900-image** version comes from the HuggingFace
> mirror (route B, ~9.5 GB) or the original Mendeley download (route C).
> Details: [docs/dataset_research.md](docs/dataset_research.md) §6.

## 10. Candidate Models for the Next Phase (research only — nothing trained yet)

| Type | Models to try later | Reference |
|---|---|---|
| Classical ML (hand-made features) | SVM, Random Forest, K-Nearest Neighbors | [docs/model_research.md](docs/model_research.md) |
| Deep learning | Custom CNN, VGG, ResNet, MobileNet, EfficientNet | [docs/model_research.md](docs/model_research.md) |
| Why transfer learning? | 6,900 images is modest for 80 classes → pretrained ImageNet backbones (MobileNetV2 / ResNet50 / EfficientNetB0) train fast and generalize better | [docs/model_research.md](docs/model_research.md) |

## 11. Project Structure

```
medicinal-plant-classification/
│
├── README.md                     <- you are here (problem understanding)
│
├── data/
│   ├── raw/                      <- downloaded dataset (read-only)
│   ├── processed/                <- cleaned data (later phase)
│   └── README.md                 <- HOW TO DOWNLOAD THE DATASET
│
├── docs/
│   ├── dataset_research.md       <- dataset candidates + comparison + verification
│   ├── model_research.md         <- ML/DL model comparison (no training)
│   └── project_pipeline.md       <- future image-classification pipeline
│
├── scripts/
│   ├── download_dataset.py       <- downloads data (Kaggle / HuggingFace / manual)
│   └── explore_dataset.py        <- read-only dataset inspection → reports/
│
├── reports/
│   └── dataset_summary.txt       <- generated by explore_dataset.py
│
├── requirements.txt
└── .gitignore
```

## 12. Getting Started

```powershell
# 0) Python 3.10+ (validated on 3.12)
python --version

# 1) Install dependencies (Pillow + matplotlib only)
python -m pip install -r requirements.txt

# 2) Download the dataset - pick ONE route (details in data/README.md)
python scripts/download_dataset.py --source kaggle   # ~265 MB, NO login needed (verified 2026-09-30)
python scripts/download_dataset.py --source hf       # ~9.5 GB, full 80-class version, no login
python scripts/download_dataset.py --source manual   # prints manual download steps

# 3) Inspect & verify the dataset (read-only; writes reports/)
python scripts/explore_dataset.py
#    -> reports/dataset_summary.txt      (already generated: 40 classes / 5,945 images)
#    -> reports/class_distribution.png

# 4) Verify connectivity/credentials without downloading anything
python scripts/download_dataset.py --source check
```

## 13. Documentation Map

| Question | Document |
|---|---|
| What is the problem & why classification? | this README |
| Which dataset, where from, is it legit? | [docs/dataset_research.md](docs/dataset_research.md) |
| How do I get the data? | [data/README.md](data/README.md) |
| Which models could we use later? | [docs/model_research.md](docs/model_research.md) |
| What will the future pipeline look like? | [docs/project_pipeline.md](docs/project_pipeline.md) |

## 14. Verified Facts vs. Assumptions

**Verified online from primary sources (2026-09-30)** — dataset identity, license (CC BY 4.0),
DOI, published class count (80), published image count (6,900) with per-class counts summing
to 6,900, sampled resolutions, paper metadata. Each is sourced in
[docs/dataset_research.md](docs/dataset_research.md).

**Verified locally on this machine (downloaded + inspected)** —
Kaggle mirror: 264,918,525-byte zip, **40 classes, 5,945 JPEGs, 0 corrupt files,
57 exact-duplicate groups, 15 distinct sizes (277–600 px), 252.4 MB**, folder nesting
`.../Medicinal plant dataset/<Class>/*.jpg`. Full evidence:
`reports/dataset_summary.txt` + `reports/class_distribution.png`.

**Known discrepancy (documented, not hidden)** — Kaggle mirror description claims 80
classes but ships 40; the full 80/6,900 set is obtained via route B or C.

**Still pending / assumptions** — exact contents of the Mendeley original (route C not run),
local inspection of the HF parquet shards (route B not run), near-duplicate (non-identical)
images — visual deduplication is a future-phase task.

## 15. Citation & License (required when using the data)

The dataset is **CC BY 4.0**: you may use it for study and research **if you credit the authors**.

```text
B R, Pushpa; Rani, Shobha (2023), "Indian Medicinal Leaves Image Datasets",
Mendeley Data, V3, doi: 10.17632/748f8jkphb.3

Paper: Pushpa B R., Rani N. Shobha (2023). "DIMPSAR: Dataset for Indian medicinal
plant species analysis and recognition." Data in Brief 49, 109388.
https://doi.org/10.1016/j.dib.2023.109388
```

## 16. Next Phase (NOT done yet — awaiting instruction)

1. Download + locally verify the dataset (this repo's scripts are ready).
2. Data cleaning & stratified train/validation/test split → `data/processed/`.
3. Baselines: SVM / Random Forest / KNN on hand-crafted features.
4. Deep learning: Custom CNN + transfer learning (MobileNetV2, ResNet50, EfficientNetB0).
5. Evaluation: accuracy, macro-F1 (because of class imbalance), confusion matrix.
6. *(Only if asked)* a simple inference demo — no web app planned yet.

**Constraints honored:** no model training, no web app, no UI, no fabricated data or statistics.


