# data/ — where the dataset lives

**Selected dataset:** *Indian Medicinal Leaves Image Datasets* (a.k.a. **DIMPSAR**)
— 80 classes, 6,900 leaf images, license **CC BY 4.0**.
Full research & verification: [../docs/dataset_research.md](../docs/dataset_research.md)

```
data/
├── raw/         <- dataset EXACTLY as downloaded (READ-ONLY, never modified)
├── processed/   <- cleaned/resized/split data (created in a LATER phase)
└── README.md    <- this file
```

## Rules

1. `data/raw/` is **read-only**. Scripts only open files here for reading.
2. Anything derived (cleaning, resizing, splitting) goes to `data/processed/` later.
3. **Never commit images to git** — `.gitignore` already excludes `data/raw/*` and
   `data/processed/*`. Share the *source links*, not the files.
4. When publishing anything based on this project, **cite the dataset** (see bottom).

## Download route A — script via Kaggle API (recommended, ~265 MB)

```powershell
python scripts/download_dataset.py --source kaggle
```

* Uses the official Kaggle API endpoint for the mirror
  `warcoder/indian-medicinal-plant-image-dataset` (CC BY 4.0).
* **No login was required when this was verified (2026-09-30)** — the endpoint served
  the public zip anonymously (HTTP 206, `application/zip`, 264,918,525 bytes).
* If Kaggle ever starts requiring authentication, the script prints instructions:
  create a **free** account → profile → *Settings → API → Create New API Token* →
  save the downloaded `kaggle.json` to `~/.kaggle/`
  (Windows: `C:\Users\<you>\.kaggle\kaggle.json`), or set the environment variables
  `KAGGLE_USERNAME` / `KAGGLE_KEY`. **Never commit those credentials** (`.gitignore`
  already covers `kaggle.json`).
* Resulting layout — **verified locally (2026-09-30)**:

```
data/raw/indian_medicinal_leaves_kaggle/
        └── Medicinal plant dataset/        <- verified nesting (1 wrapper folder)
                └── <Class Name>/            e.g. Neem/, Tulasi/, Brahmi/, ...
                        └── *.jpg
```

* **Verified local result:** 40 classes · **5,945 JPEGs** · 252.4 MB · 0 corrupt ·
  115–187 images/class (balanced) · **57 exact-duplicate groups** (remove before
  splitting) · full evidence in `reports/dataset_summary.txt`.
* ⚠️ The mirror's *description* says 80 classes, but the archive ships **40** —
  use route B or C if you need the full 80-class / 6,900-image version.

## Download route B — HuggingFace (no login at all, ~9.5 GB)

```powershell
python scripts/download_dataset.py --source hf
```

* Mirrors the same dataset as 20 parquet shards:
  `Project-AgML/DIMPSAR_medicinal_leaf_classification` (CC BY 4.0).
* Saves to `data/raw/hf_parquet/train-00000-of-00020.parquet` … `-00019`.
* Much bigger on disk than route A (images re-encoded inside parquet).
* Inspecting parquet requires `pip install pyarrow` (see `requirements.txt`).

## Download route C — manual (original Mendeley source)

```powershell
python scripts/download_dataset.py --source manual   # prints these steps
```

1. Open the original record: https://data.mendeley.com/datasets/748f8jkphb/3
   (DOI `10.17632/748f8jkphb.3`, V3, CC BY 4.0).
2. Sign in (free) if asked → **Download All Files**.
3. Unzip and place the folder(s) anywhere **inside `data/raw/`**, keeping the
   per-species class folders intact, e.g.:

```
data/raw/Indian Medicinal Leaves/Neem/IMG_001.jpg
data/raw/Indian Medicinal Leaves/Tulsi/IMG_047.jpg
```

A browser download from the Kaggle page (route A's web UI) also works —
just unzip into `data/raw/`.

## After downloading — verify (mandatory step)

```powershell
python scripts/explore_dataset.py
```

This **read-only** inspection writes `reports/dataset_summary.txt` and
`reports/class_distribution.png`, and checks the download against the verified
expectation (**80 classes / 6,900 images**). If it reports a difference, fix the
download before training (partial extraction, wrong folder, extra files…).

Quick pre-flight (no data transferred):

```powershell
python scripts/download_dataset.py --source check
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Kaggle API returned HTTP 401/403` | Kaggle now wants credentials → add `kaggle.json` (route A notes) or use route B/C |
| `HTTP 404` | Mirror renamed → use route B or C |
| Network/SSL errors | Check internet/proxy; try route C (browser) |
| explore says *no images found* | Unzip did not happen or files landed outside `data/raw/` |
| parquet mode needs pyarrow | `pip install pyarrow` |
| Disk space | route A ≈ 0.3 GB · route B ≈ 9.5 GB |

## Citation & license (CC BY 4.0 — attribution is mandatory)

```text
B R, Pushpa; Rani, Shobha (2023), "Indian Medicinal Leaves Image Datasets",
Mendeley Data, V3, doi: 10.17632/748f8jkphb.3

Paper: Pushpa B R., Rani N. Shobha (2023). "DIMPSAR: Dataset for Indian medicinal
plant species analysis and recognition." Data in Brief 49, 109388.
https://doi.org/10.1016/j.dib.2023.109388
```


