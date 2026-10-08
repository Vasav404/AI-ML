# Dataset Research — Medicinal Plant Leaf Images

**Researched & verified on: 2026-09-30**
**Goal:** find a *publicly available, academically usable* dataset of **medicinal plant leaves**
with many classes, clear labels and enough images per class — then verify its facts from
primary sources (not blogs).

---

## 1. How the research was done (reproducible)

| Source queried | What it gave us |
|---|---|
| Kaggle public dataset API (`/api/v1/datasets/list`, `/datasets/view`) | candidate search, exact byte sizes, licenses, download counts, descriptions |
| Mendeley Data records + **DataCite DOI API** (`api.datacite.org/dois/...`) | authoritative title, authors, **license (CC BY 4.0)**, DOI, publication year |
| **Crossref API** | companion paper metadata (journal, volume, pages, DOI) |
| HuggingFace Hub **API** + **datasets-server API** | exact class count, image count (6,900), per-class counts, sampled image resolutions |

Legend: ✅ = verified from a primary source on 2026-09-30 · ⚠️ = inconsistency/risk noted ·
❌ = *not verifiable without downloading — will be checked locally by `scripts/explore_dataset.py`*.
**No number in this document is invented.**

## 2. Requirements used to judge candidates

1. Medicinal plant **leaf** images (not generic crops/diseases)
2. Many species classes; sufficient images per class
3. Clear labels (folder-per-class or CSV)
4. Good image quality; publicly downloadable
5. License permitting **academic/research use** (Creative Commons preferred)
6. Preferably an accompanying research paper or institutional source

## 3. Candidates

### ✅ CANDIDATE 1 — SELECTED: *Indian Medicinal Leaves Image Datasets* (paper name: **DIMPSAR**)

| Field | Finding |
|---|---|
| **1. Name** | Indian Medicinal Leaves Image Datasets — known by its paper name **DIMPSAR** |
| **2. Source URLs** | **Original:** https://data.mendeley.com/datasets/748f8jkphb/3 — DOI **10.17632/748f8jkphb.3** (V3, 5 May 2023) ✅<br>**Kaggle mirror:** https://www.kaggle.com/datasets/warcoder/indian-medicinal-plant-image-dataset ✅<br>**HuggingFace mirror:** https://huggingface.co/datasets/Project-AgML/DIMPSAR_medicinal_leaf_classification ✅ |
| **3. Number of classes** | **80** ✅ — Kaggle description: "80 distinct Indian leaf varieties"; HF card lists exactly 80 class names; HF statistics API: `n_unique = 80` |
| **4. Number of images** | **6,900** ✅ — HF card: "6,900 images across 80 classes"; HF `dataset_info.splits[0].num_examples = 6900` |
| **5. Image format** | JPEG ✅ on HF mirror (`image.jpg` in row URLs). Kaggle/Mendeley mirror format ❌ pending local check |
| **6. Resolution** | Mixed ✅ (HF mirror): first 100 rows are all **600×450**; HF statistics (partial): min **450**, median **450**, max **4656** px. Mendeley abstract: images "captured with varying background without any environment constraints" |
| **7. Labels provided** | ✅ Yes — one folder per species (image mirrors); integer `ClassLabel` 0–79 (HF parquet) |
| **8. Specifically medicinal plants?** | ✅ Yes — Indian medicinal species (Tulsi, Neem, Bhrami, Henna, Parijatha, Nelavembu, …). Caveat: the 80 classes also include common kitchen/garden leaves (Beans, Onion, Tomato, Mango, …) — a typical "medicinal + household herb" collection |
| **9. License** | **CC BY 4.0** ✅ — DataCite `rightsList` for the DOI; Kaggle mirror: CC BY 4.0; HF card: `cc-by-4.0` |
| **10. Advantages** | Medicinal-specific & large (80 classes); verified counts; peer-reviewed companion paper (*Data in Brief*, Elsevier); three download routes incl. no-login HuggingFace; per-class counts published; indexed by AgML research library |
| **11. Limitations** | Strong class imbalance (**8–177**/class); two classes have only **8** images; mixed resolutions (450→4656 px); uncontrolled backgrounds (realistic but harder); no official train/val/test split; possible near-duplicate photos; label-name typos ("Coriender", "Pomoegranate", "Raddish", "Spinach1" — cosmetic); Kaggle mirror 265 MB vs HF 9.5 GB → mirrors may differ in resolution/compression (verify locally) |
| **12. Suitable for this project?** | ✅ **YES — SELECTED** (rationale in §4) |

**Companion paper (verified via Crossref):**
Pushpa B R., Rani N. Shobha (2023). *DIMPSAR: Dataset for Indian medicinal plant species
analysis and recognition.* **Data in Brief 49, 109388** — https://doi.org/10.1016/j.dib.2023.109388

**✅ Verified per-class image counts (HF dataset card; corroborated by HF statistics API):**

```text
Aloevera:118   Amla:67        Amruthaballi:91  Arali:89       Astma_weed:82   Badipala:76
Balloon_Vine:61 Bamboo:118    Beans:97         Betel:114      Bhrami:104      Bringaraja:73
Caricature:76  Castor:129     Catharanthus:134 Chakte:68      Chilly:69       Citron lime (herelikai):99
Coffee:83      Common rue(naagdalli):67        Coriender:115  Curry:168       Doddpathre:142
Drumstick:56   Ekka:81        Eucalyptus:80    Ganigale:75    Ganike:63       Gasagase:79
Ginger:82      Globe Amarnath:81               Guava:128      Henna:80        Hibiscus:118
Honge:113      Insulin:89     Jackfruit:110    Jasmine:49     Kambajala:59    Kasambruga:48
Kohlrabi:73    Lantana:76     Lemon:123        Lemongrass:8   Malabar_Nut:51  Malabar_Spinach:79
Mango:103      Marigold:93    Mint:135         Neem:132       Nelavembu:90    Nerale:62
Nooni:72       Onion:92       Padri:73         Palak(Spinach):149           Papaya:135
Parijatha:66   Pea:47         Pepper:8         Pomoegranate:75               Pumpkin:92
Raddish:40     Rose:106       Sampige:61       Sapota:44      Seethaashoka:47 Seethapala:114
Spinach1:67    Tamarind:176   Taro:69          Tecoma:69      Thumbe:74       Tomato:62
Tulsi:177      Turmeric:39    ashoka:81        camphor:66     kamakasturi:67  kepala:76
```

Smallest classes → split/augmentation will matter: **Lemongrass 8, Pepper 8, Turmeric 39,
Raddish 40, Sapota 44, Pea 47, Seethaashoka 47, Kasambruga 48, Jasmine 49**.

---

### ✅ CANDIDATE 2: *Medicinal Leaf Dataset* (Roopashree & Anitha, Mendeley)

| Field | Finding |
|---|---|
| **1. Name** | Medicinal Leaf Dataset |
| **2. Source URL** | https://data.mendeley.com/datasets/nnytj2v3n5/1 — DOI **10.17632/nnytj2v3n5.1** (V1, 22 Oct 2020) ✅ |
| **3. Classes** | ⚠️ **official description contradicts itself**: "comprises of **thirty** species …" but also "consists of 1500 images of **forty** species" (both quotes verbatim from the Mendeley record) — treat as ~30–40 until downloaded |
| **4. Images** | **1,500** ✅ ("60 to 100 high-quality images" per species, verbatim) |
| **5. Format** | ❌ not stated on the record (pending download) |
| **6. Resolution** | ❌ not stated ✅ capture gear stated: Samsung S9+ phone camera + Canon inkjet printer; "slightly rotated and tilted" images; healthy mature leaves |
| **7. Labels** | ✅ folders named by botanical/scientific name (verbatim: "folders are named as per the species botanical/scientific name") |
| **8. Medicinal?** | ✅ explicitly Ayurvedic/medicinal herbs (e.g. *Santalum album*, *Muntingia calabura*, *Plectranthus amboinicus*) |
| **9. License** | **CC BY 4.0** ✅ (DataCite `rightsList`, open access) |
| **10. Advantages** | Clean controlled capture; small & quick to download; CC BY 4.0; cited works (DataCite: 12 citations) |
| **11. Limitations** | Only 1,500 images for 30–40 classes (~37–50/class) → thin for deep learning; official species-count inconsistency; no verified automatic-download mirror found |
| **12. Suitable?** | ⚠️ Backup candidate only — good for a tiny demo, weaker for the main 80-class project |

### ⚠️ CANDIDATE 3: *MedLeaves — Medicinal Plant Leaves Dataset* (Kaggle `mirlab`)

| Field | Finding |
|---|---|
| **1. Name** | MedLeaves — Medicinal Plant Leaves Dataset |
| **2. Source URL** | https://www.kaggle.com/datasets/mirlab/medleaves-medicinal-plant-leaves-dataset ✅ |
| **3. Classes** | **30** ✅ (subtitle: "…Medicinal Plant Leaves - 30 classes") |
| **4. Images** | count ❌ not published; archive size **1,944,889,745 bytes (≈1.94 GB)** ✅ Kaggle API; contains a YOLO-format detection set **plus** a "Cropped Dataset" for DenseNet classification (per description) |
| **5. Format** | detection images + YOLO labels + cropped classification images (per description); exact formats ❌ pending download |
| **6. Resolution** | ❌ pending download |
| **7. Labels** | ✅ class folders **and** YOLO bounding boxes (usable for detection later) |
| **8. Medicinal?** | ✅ yes |
| **9. License** | ⚠️ **"Unknown"** on Kaggle (verified via API) — risky for redistribution; dataset asks for a citation: Sharafudeen et al., *Computers in Biology and Medicine* 174 (2024): 108467 (per dataset description) |
| **10. Advantages** | Recent (2024) with peer-reviewed paper; multi-task (detection + classification); ~1,000 downloads |
| **11. Limitations** | **Unclear license**; class names/counts per class not on the card ❌; heavy (1.94 GB); requires Kaggle login |
| **12. Suitable?** | ❌ Not selected — license not verified (project rule: license must permit academic use *clearly*) |

### ⚠️ CANDIDATE 4: *Medicinal Leaf Health Classification* (Kaggle `jocelyndumlao`)

| Field | Finding |
|---|---|
| **1. Name** | Medicinal Leaf Health Classification — subtitle: "A Diverse Medicinal Plant Leaf Dataset for AI-Based Health Condition Detection" |
| **2. Source URL** | https://www.kaggle.com/datasets/jocelyndumlao/medicinal-leaf-health-classification ✅ |
| **3. Classes** | ❌ not stated anywhere in the metadata (class count unknown) |
| **4. Images** | ❌ count not stated; archive size **8,624,343,823 bytes (≈8.6 GB)** ✅ Kaggle API |
| **5–6. Format/resolution** | ❌ unknown (empty description) |
| **7. Labels** | unknown ❌ (task tag = image classification, 9 public notebooks) |
| **8. Medicinal?** | ✅ medicinal leaves, but target is **health-condition detection**, not species classification |
| **9. License** | **CC0 (Public Domain)** ✅ Kaggle API |
| **10. Advantages** | CC0 license; recent (May 2026); some notebooks exist |
| **11. Limitations** | **No description, no counts, no paper, unclear provenance/collection method**; 8.6 GB; only 94 downloads → too little community validation |
| **12. Suitable?** | ❌ Not selected — insufficient verifiable information (project rule: do not invent statistics) |

### ❌ CANDIDATE 5 (rejected — not medicinal): *PlantVillage* mirror (Kaggle `emmarex/plantdisease`)

| Field | Finding |
|---|---|
| **1–2. Name/URL** | "PlantVillage Dataset" — https://www.kaggle.com/datasets/emmarex/plantdisease ✅ (mirror of spMohanty's PlantVillage GitHub) |
| **3–4. Size** | archive **342,230,723 bytes (≈342 MB)** ✅, 179,349 downloads ✅ |
| **8. Medicinal?** | ❌ **No** — it classifies **crop diseases** (e.g. tomato blight), not medicinal species |
| **9. License** | ⚠️ "Unknown" on Kaggle (verified via API) |
| **12. Suitable?** | ❌ Rejected — kept here only as the famous *plant* benchmark people find first |

---

## 4. Comparison table

| # | Dataset | Classes | Images | Medicinal? | License | Auto-download | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | **Indian Medicinal Leaves (DIMPSAR)** — Mendeley + Kaggle + HF | **80** ✅ (published card) · **40** ✅ (local Kaggle mirror) | **6,900** ✅ (card) · **5,945** ✅ (local) | ✅ yes | **CC BY 4.0** ✅ | Kaggle API (no login verified) / HF / manual | ✅ **SELECTED** — local mirror downloaded & inspected (§6) |
| 2 | Medicinal Leaf Dataset (Mendeley, 2020) | ~30–40 ⚠️ (official text inconsistent) | 1,500 ✅ | ✅ yes | CC BY 4.0 ✅ | manual (Mendeley) | backup |
| 3 | MedLeaves (Kaggle `mirlab`, 2024) | 30 ✅ | ≈1.94 GB archive (count ❌) | ✅ yes | ⚠️ "Unknown" | Kaggle API | rejected (license) |
| 4 | Medicinal Leaf Health Classification (Kaggle, 2026) | ❌ unknown | ≈8.6 GB (count ❌) | ✅ yes (health target) | CC0 ✅ | Kaggle API | rejected (info gaps) |
| 5 | PlantVillage mirror (Kaggle) | crop-disease classes | ≈342 MB | ❌ no | ⚠️ "Unknown" | Kaggle API | rejected (wrong topic) |

## 5. Why candidate 1 was chosen (not blindly the first result)

1. **Most classes** among medicinal candidates (80 vs 30 vs unknown) → a genuinely challenging
   80-way problem, good for a serious college DL project.
2. **Fully verifiable**: counts (6,900 / 80), license (CC BY 4.0), DOI and companion paper all
   confirmed from *primary* APIs — candidate 4 has no verifiable counts; candidate 3 has an unclear license.
3. **Clear labels** (folder-per-class / ClassLabel) and published per-class counts.
4. **Multiple legal access routes**, including a **no-login** HuggingFace mirror and a small
   (~265 MB) Kaggle mirror — reproducible for any classmate.
5. **Peer-reviewed provenance** (*Data in Brief*, Elsevier, CC BY 4.0) beats a random upload.
6. Candidate 2 is honest but **too small** (1,500 images) for 30–40 classes of deep learning.

## 6. Local download & inspection results (Kaggle mirror — verified 2026-09-30)

The Kaggle route was **actually executed on this machine** and inspected with
`scripts/explore_dataset.py` (full report: `reports/dataset_summary.txt`,
chart: `reports/class_distribution.png`). Everything in this table is **locally measured**:

| Property | Local value (Kaggle mirror) |
|---|---|
| Download | 264,918,525-byte zip from the official Kaggle API endpoint — **no login required** (verified) |
| Extracted size | 252.4 MB |
| Structure | `data/raw/indian_medicinal_leaves_kaggle/Medicinal plant dataset/<Class>/*.jpg` |
| Classes | **40** |
| Images | **5,945** — 100% readable JPEG, **0 corrupt**, 0 unlabeled, 0 non-image files |
| Images/class | min 115 (Ganike) – median 146 – mean 148.6 – max 187 (Jasmine); ratio only 1.6 → **well balanced** |
| Resolution | 15 distinct sizes; width 277–600 (median 600), height 270–600 (median 450); most common 600×450 (2,958 imgs) and 600×283 (2,279 imgs) |
| Format | **JPEG 100%** (Pillow-detected, extension `.jpg`) |
| Exact duplicates | **57 groups** (57 redundant files) → remove before train/test splitting |
| Class names | slightly different spelling than the 80-class card: `Tulasi`, `Brahmi`, `Gauva`, `Amruta_Balli`, `Nagadali`, … |

### ⚠️ Verified mirror discrepancy (important!)

* The Kaggle mirror's **description claims** "80 distinct Indian leaf varieties",
  but the archive it actually serves contains **40 class folders / 5,945 images** —
  confirmed twice: by the folder listing and by the independent inspection script.
* The HuggingFace mirror's metadata states **80 classes / 6,900 images**; its 80
  published per-class counts sum to **exactly 6,900** (computed), and its class names
  do **not** match the Kaggle folder names → the two mirrors ship **different snapshots**
  of the same credited source (Mendeley V3 / DIMPSAR paper).
* **Practical conclusion:**
  * local Kaggle copy = ready-to-use, balanced **40-class** problem (great for this project),
  * full **80-class / 6,900-image** version = route B (HuggingFace) or route C (Mendeley original).
* Both numbers come from primary sources (HuggingFace API) or direct local measurement —
  nothing was invented.

---

## 7. Verified vs. still-to-be-verified

| Verified (online **and/or** locally) | Still to verify |
|---|---|
| Identity, DOI, authors, year (DataCite/Crossref) | Mendeley original's exact file list/counts (route C not run) |
| License CC BY 4.0 (3 independent sources) | whether Mendeley original == HF mirror image-for-image |
| 80 classes on the published card (+ full class list) | local inspection of the HF parquet shards (route B not run) |
| 6,900 images on the published card (+ per-class counts summing to 6,900) | — |
| **Local Kaggle mirror: 40 classes / 5,945 JPEGs / 0 corrupt / 57 duplicate groups** | — |
| **Local resolutions: 277×600 px range, 15 distinct sizes** | — |
| **Local folder nesting** (`…/Medicinal plant dataset/<Class>/*.jpg`) | — |
| **Mirror discrepancy: Kaggle 40/5,945 ≠ card 80/6,900** | — |

## 8. License obligations & citation

**CC BY 4.0** allows sharing and adaptation — including academic projects — **provided
attribution is given**. Suggested citation (also in the root README):

```text
B R, Pushpa; Rani, Shobha (2023), "Indian Medicinal Leaves Image Datasets",
Mendeley Data, V3, doi: 10.17632/748f8jkphb.3

Paper: Pushpa B R., Rani N. Shobha (2023). "DIMPSAR: Dataset for Indian medicinal plant
species analysis and recognition." Data in Brief 49, 109388.
https://doi.org/10.1016/j.dib.2023.109388
```

Do **not** redistribute the raw images inside this git repository (see `.gitignore`) —
link people to the official sources instead.

## 9. Appendix — exact verification endpoints used (re-runnable)

```text
https://api.datacite.org/dois/10.17632/748f8jkphb.3                  (license/DOI/authors)
https://api.datacite.org/dois/10.17632/nnytj2v3n5.1                  (candidate 2)
https://api.crossref.org/works?query.bibliographic=DIMPSAR+Indian+medicinal+plant
https://www.kaggle.com/api/v1/datasets/view/warcoder/indian-medicinal-plant-image-dataset
https://www.kaggle.com/api/v1/datasets/view/mirlab/medleaves-medicinal-plant-leaves-dataset
https://www.kaggle.com/api/v1/datasets/view/jocelyndumlao/medicinal-leaf-health-classification
https://www.kaggle.com/api/v1/datasets/view/emmarex/plantdisease
https://huggingface.co/api/datasets/Project-AgML/DIMPSAR_medicinal_leaf_classification
https://datasets-server.huggingface.co/statistics?dataset=Project-AgML%2FDIMPSAR_medicinal_leaf_classification&config=default&split=train
https://datasets-server.huggingface.co/first-rows?dataset=Project-AgML%2FDIMPSAR_medicinal_leaf_classification&config=default&split=train
```



