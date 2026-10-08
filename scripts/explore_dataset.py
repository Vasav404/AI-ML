#!/usr/bin/env python3
"""
explore_dataset.py
==================
Phase 4 - READ-ONLY inspection of the downloaded dataset.

What it reports
---------------
  * total number of images, number of classes, images per class
  * corrupt / unreadable images (Pillow verify + full pixel load)
  * image dimensions (min/max/mean/median, most common sizes)
  * image formats and file extensions
  * class distribution (+ reports/class_distribution.png bar chart)
  * exact duplicate files (SHA-256 of file content)
  * comparison against the VERIFIED expected size (80 classes / 6,900 images)

Outputs
-------
  reports/dataset_summary.txt      (overwrites the placeholder)
  reports/class_distribution.png   (unless --skip-chart)

IMPORTANT
---------
  * This script NEVER modifies anything under data/raw/ (opens files 'rb' only).
  * It writes NOTHING when no dataset is found - it never fabricates statistics.
  * Supports both image-folder downloads (Kaggle / Mendeley) and the HuggingFace
    parquet shards (requires 'pyarrow' for the parquet route).

Usage
-----
  python scripts/explore_dataset.py
  python scripts/explore_dataset.py --skip-chart --fast
  python scripts/explore_dataset.py --data-dir path/to/raw --out-dir path/to/reports
"""

from __future__ import annotations

import argparse
import hashlib
import io
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    print("[error] Pillow is required:  python -m pip install pillow", file=sys.stderr)
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw"
DEFAULT_REPORTS = PROJECT_ROOT / "reports"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".ppm", ".jfif"}
HASH_CHUNK = 1024 * 1024

# Verified expected size of the selected dataset (see docs/dataset_research.md):
# HuggingFace dataset card: "6,900 images across 80 classes".
EXPECTED_IMAGES = 6900
EXPECTED_CLASSES = 80

# Verified class names of the HuggingFace parquet mirror (label index -> name),
# taken from the dataset card on 2026-09-30 (Project-AgML/DIMPSAR_...).
CLASS_NAMES_80 = [
    "Aloevera", "Amla", "Amruthaballi", "Arali", "Astma_weed", "Badipala",
    "Balloon_Vine", "Bamboo", "Beans", "Betel", "Bhrami", "Bringaraja",
    "Caricature", "Castor", "Catharanthus", "Chakte", "Chilly",
    "Citron lime (herelikai)", "Coffee", "Common rue(naagdalli)", "Coriender",
    "Curry", "Doddpathre", "Drumstick", "Ekka", "Eucalyptus", "Ganigale",
    "Ganike", "Gasagase", "Ginger", "Globe Amarnath", "Guava", "Henna",
    "Hibiscus", "Honge", "Insulin", "Jackfruit", "Jasmine", "Kambajala",
    "Kasambruga", "Kohlrabi", "Lantana", "Lemon", "Lemongrass", "Malabar_Nut",
    "Malabar_Spinach", "Mango", "Marigold", "Mint", "Neem", "Nelavembu",
    "Nerale", "Nooni", "Onion", "Padri", "Palak(Spinach)", "Papaya",
    "Parijatha", "Pea", "Pepper", "Pomoegranate", "Pumpkin", "Raddish",
    "Rose", "Sampige", "Sapota", "Seethaashoka", "Seethapala", "Spinach1",
    "Tamarind", "Taro", "Tecoma", "Thumbe", "Tomato", "Tulsi", "Turmeric",
    "ashoka", "camphor", "kamakasturi", "kepala",
]


def say(msg: str = "", end: str = "\n") -> None:
    try:
        print(msg, flush=True, end=end)
    except OSError:
        pass  # stdout closed (e.g. piping) - never let logging kill the inspection


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def has_images_below(folder: Path) -> bool:
    for p in folder.rglob("*"):
        if is_image(p):
            return True
    return False


# ----------------------------------------------------------------------------
# Locate the "class folder" level: the directory whose immediate subdirectories
# each contain images (works for any nesting the download routes produce).
# ----------------------------------------------------------------------------
def find_dataset_root(raw: Path):
    """Return (root: Path|None, classes: dict[name -> list[Path]], loose: list[Path])."""
    if not raw.is_dir():
        return None, {}, []

    dirs = [raw] + [p for p in sorted(raw.rglob("*")) if p.is_dir()]
    candidates = []
    for d in dirs:
        rel = d.relative_to(raw)
        if any(part.startswith(".") for part in rel.parts):
            continue
        try:
            subdirs = [s for s in sorted(d.iterdir())
                       if s.is_dir() and not s.name.startswith(".")]
        except OSError:
            continue
        with_images = [s for s in subdirs if has_images_below(s)]
        if len(with_images) >= 2:
            depth = len(rel.parts)
            candidates.append((len(with_images), depth, d, with_images))

    if not candidates:
        loose = sorted(p for p in raw.rglob("*") if is_image(p))
        return (raw if loose else None), {}, loose

    # Prefer the level with the MOST class subfolders; deeper level breaks ties.
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    _, _, root, with_images = candidates[0]

    classes = {}
    for s in with_images:
        classes[s.name] = sorted(p for p in s.rglob("*") if is_image(p))

    class_set = set(with_images)
    loose = sorted(
        p for p in root.rglob("*")
        if is_image(p) and not any(c in p.parents for c in class_set)
    )
    return root, classes, loose


def inspect_one_image(path: Path, deep: bool = True) -> dict:
    """Open a file read-only; return {'ok', 'w', 'h', 'fmt', 'mode', 'bytes', 'sha', 'err'}."""
    info = {"ok": False, "w": None, "h": None, "fmt": None, "mode": None,
            "bytes": 0, "sha": None, "err": None, "ext": path.suffix.lower()}
    try:
        info["bytes"] = path.stat().st_size
        info["sha"] = sha256_of(path)
        with Image.open(path) as im:
            info["fmt"] = im.format
            info["mode"] = im.mode
            info["w"], info["h"] = im.size
            im.verify()                      # structural check (markers/CRCs)
        if deep:
            with Image.open(path) as im2:    # full pixel decode catches truncation
                im2.load()
        info["ok"] = True
    except Exception as exc:                 # noqa: BLE001 - we *want* to catch everything
        info["err"] = f"{type(exc).__name__}: {exc}"[:200]
    return info


def _new_stats(mode: str, root: Path) -> dict:
    return {
        "mode": mode,
        "root": str(root),
        "class_counts": {},                 # name -> image count
        "class_bytes": defaultdict(int),    # name -> bytes
        "total_images": 0,
        "labeled_images": 0,
        "loose_images": 0,
        "total_bytes": 0,
        "widths": [],
        "heights": [],
        "size_counter": Counter(),          # (w, h) -> n
        "formats": Counter(),
        "exts": Counter(),
        "corrupt": [],                      # (path, error)
        "nonimage_files": [],
        "hash_paths": defaultdict(list),    # sha -> [paths]  (for duplicates)
        "checked_files": 0,
    }


def _record(stats: dict, label: str | None, info: dict, rel: str) -> None:
    stats["checked_files"] += 1
    stats["total_bytes"] += info["bytes"]
    if info["sha"]:
        stats["hash_paths"][info["sha"]].append(rel)
    if not info["ok"]:
        stats["corrupt"].append((rel, info["err"] or "unknown error"))
        return
    stats["total_images"] += 1
    if label is None:
        stats["loose_images"] += 1
    else:
        stats["labeled_images"] += 1
        stats["class_bytes"][label] += info["bytes"]
    stats["widths"].append(info["w"])
    stats["heights"].append(info["h"])
    stats["size_counter"][(info["w"], info["h"])] += 1
    stats["formats"][info["fmt"] or "UNKNOWN"] += 1
    stats["exts"][info["ext"] or "none"] += 1


def inspect_folder(root: Path, classes: dict, loose: list, deep: bool,
                   base: Path) -> dict:
    """Inspect an image-folder dataset. READ-ONLY."""
    stats = _new_stats("image folders", root)
    tasks = []
    for name in sorted(classes):
        for p in classes[name]:
            tasks.append((name, p))
    for p in loose:
        tasks.append((None, p))

    say(f"Scanning {len(tasks)} image files (read-only, deep={deep})...")
    for i, (name, p) in enumerate(tasks, start=1):
        info = inspect_one_image(p, deep=deep)
        rel = str(p.relative_to(base)) if p.is_relative_to(base) else str(p)
        _record(stats, name, info, rel)
        if name is not None:
            stats["class_counts"][name] = stats["class_counts"].get(name, 0) \
                + (1 if info["ok"] else 0)
        if i % 500 == 0 or i == len(tasks):
            say(f"\r  {i}/{len(tasks)} files checked...", end="")
    say()

    # non-image files under the class root (ignore hidden)
    for p in sorted(root.rglob("*")):
        if p.is_file() and not is_image(p) and not p.name.startswith(".") \
                and p.suffix.lower() != ".parquet":
            stats["nonimage_files"].append(
                str(p.relative_to(base)) if p.is_relative_to(base) else str(p))
    return stats


# ----------------------------------------------------------------------------
# HuggingFace parquet shards (read-only; needs 'pyarrow')
# ----------------------------------------------------------------------------
def inspect_parquet(files: list[Path], deep: bool, base: Path) -> dict | None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        say("[error] pyarrow is not installed - cannot inspect parquet shards.")
        say("        pip install pyarrow     (or download image folders via '--source kaggle')")
        return None

    stats = _new_stats("huggingface parquet", files[0].parent)
    total_rows = 0
    for f in files:
        try:
            pf = pq.ParquetFile(f)
            total_rows += pf.metadata.num_rows
        except Exception as exc:  # noqa: BLE001
            say(f"[warn] cannot read {f.name}: {exc}")
    say(f"Inspecting {len(files)} parquet shard(s), {total_rows} rows total "
        f"(read-only, deep={deep})...")

    done = 0
    for f in files:
        try:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=256, columns=["label", "image"]):
                data = batch.to_pydict()
                labels = data["label"]
                images = data["image"]
                for lab, im in zip(labels, images):
                    done += 1
                    rel = f"{f.name}#row{done}"
                    info = {"ok": False, "w": None, "h": None, "fmt": None,
                            "mode": None, "bytes": 0, "sha": None, "err": None,
                            "ext": ".parquet"}
                    try:
                        raw_bytes = im.get("bytes") if isinstance(im, dict) else im
                        if raw_bytes is None:
                            raise ValueError("image entry has no bytes")
                        info["bytes"] = len(raw_bytes)
                        info["sha"] = hashlib.sha256(raw_bytes).hexdigest()
                        with Image.open(io.BytesIO(raw_bytes)) as img:
                            info["fmt"] = img.format
                            info["mode"] = img.mode
                            info["w"], info["h"] = img.size
                            if deep:
                                img.load()
                            else:
                                img.verify()
                        info["ok"] = True
                    except Exception as exc:  # noqa: BLE001
                        info["err"] = f"{type(exc).__name__}: {exc}"[:200]

                    name = None
                    if isinstance(lab, int) and 0 <= lab < len(CLASS_NAMES_80):
                        name = CLASS_NAMES_80[lab]
                    elif lab is not None:
                        name = str(lab)
                    _record(stats, name, info, rel)
                    if name:
                        stats["class_counts"][name] = stats["class_counts"].get(name, 0) \
                            + (1 if info["ok"] else 0)
                say(f"\r  {done}/{total_rows} rows checked...", end="")
        except Exception as exc:  # noqa: BLE001
            say(f"\n[warn] error while reading {f.name}: {exc}")
    say()
    stats["exts"] = Counter({"embedded-in-parquet": stats["total_images"]})
    return stats


# ----------------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------------
def hbytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def _median(vals: list) -> str:
    return f"{statistics.median(vals):.0f}" if vals else "n/a"


def _mean(vals: list) -> str:
    return f"{statistics.fmean(vals):.1f}" if vals else "n/a"


def build_report(stats: dict, args, data_dir: Path) -> str:
    cc: dict = stats["class_counts"]
    counts = sorted(cc.values())
    dup_groups = [paths for paths in stats["hash_paths"].values() if len(paths) > 1]
    redundant = sum(len(g) - 1 for g in dup_groups)
    L = []
    add = L.append

    add("=" * 78)
    add(" DATASET SUMMARY - generated by scripts/explore_dataset.py")
    add("=" * 78)
    add(f"Generated            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"Python               : {sys.version.split()[0]}")
    add(f"Data directory       : {data_dir}")
    add(f"Class-folder root    : {stats['root']}")
    add(f"Inspection mode      : {stats['mode']} "
        f"(deep pixel check: {not args.fast})")
    add("READ-ONLY guarantee  : source data was opened for reading only; nothing modified.")
    add("")
    add("-" * 78)
    add("1. SIZE")
    add("-" * 78)
    add(f"Files checked        : {stats['checked_files']}")
    add(f"Valid images         : {stats['total_images']}")
    add(f"Corrupt/unreadable   : {len(stats['corrupt'])}")
    add(f"Labeled images       : {stats['labeled_images']}")
    add(f"Loose images (no class folder): {stats['loose_images']}")
    add(f"Total size on disk   : {hbytes(stats['total_bytes'])} "
        f"({stats['total_bytes'] / 1e9:.3f} GB)")
    add(f"Non-image files found: {len(stats['nonimage_files'])}")
    for p in stats["nonimage_files"][:10]:
        add(f"    - {p}")
    if len(stats["nonimage_files"]) > 10:
        add(f"    ... and {len(stats['nonimage_files']) - 10} more")
    add("")
    add("-" * 78)
    add("2. CLASSES")
    add("-" * 78)
    add(f"Number of classes    : {len(cc)}")
    if counts:
        add(f"Images per class     : min {min(counts)} | median {_median(counts)} | "
            f"mean {_mean(counts)} | max {max(counts)}")
        add("")
        add("Class distribution (name : images), sorted:")
        for name in sorted(cc, key=lambda k: (-cc[k], k)):
            bar = "#" * min(60, round(cc[name] / max(max(counts), 1) * 60))
            add(f"  {name:<32} : {cc[name]:>5}  {bar}")
    else:
        add("No class folders detected (flat layout - images counted globally).")
    add("")
    add("-" * 78)
    add("3. IMAGE DIMENSIONS")
    add("-" * 78)
    if stats["widths"]:
        add(f"Width  min/mean/median/max : {min(stats['widths'])} / "
            f"{_mean(stats['widths'])} / {_median(stats['widths'])} / {max(stats['widths'])}")
        add(f"Height min/mean/median/max : {min(stats['heights'])} / "
            f"{_mean(stats['heights'])} / {_median(stats['heights'])} / {max(stats['heights'])}")
        add(f"Distinct (w x h) sizes     : {len(stats['size_counter'])}")
        add("Most common sizes:")
        for (w, h), n in stats["size_counter"].most_common(8):
            add(f"    {w} x {h:<6} : {n} images")
    else:
        add("No readable images - dimension stats unavailable.")
    add("")
    add("-" * 78)
    add("4. FORMATS")
    add("-" * 78)
    add("Actual formats (detected by Pillow):")
    for fmt, n in stats["formats"].most_common():
        add(f"    {fmt:<10} : {n}")
    add("File extensions:")
    for ext, n in stats["exts"].most_common():
        add(f"    {ext:<10} : {n}")
    add("")
    add("-" * 78)
    add("5. EXACT DUPLICATE FILES (SHA-256 of file content)")
    add("-" * 78)
    add(f"Duplicate groups     : {len(dup_groups)}")
    add(f"Redundant files      : {redundant}")
    for g in dup_groups[:args.max_list]:
        add("  group:")
        for p in g:
            add(f"    - {p}")
    if len(dup_groups) > args.max_list:
        add(f"  ... and {len(dup_groups) - args.max_list} more groups "
            f"(raise --max-list to see them)")
    add("")
    add("-" * 78)
    add("6. CORRUPT / UNREADABLE FILES")
    add("-" * 78)
    add(f"Count                : {len(stats['corrupt'])}")
    for p, err in stats["corrupt"][:args.max_list]:
        add(f"    - {p}")
        add(f"        {err}")
    if len(stats["corrupt"]) > args.max_list:
        add(f"    ... and {len(stats['corrupt']) - args.max_list} more")
    add("")
    add("-" * 78)
    add("7. FOUND vs EXPECTED (expected values verified from the dataset card)")
    add("-" * 78)
    add(f"Images    : found {stats['total_images']} | expected {EXPECTED_IMAGES} "
        f"| {'MATCH' if stats['total_images'] == EXPECTED_IMAGES else 'DIFFERENCE'}")
    add(f"Classes   : found {len(cc)} | expected {EXPECTED_CLASSES} "
        f"| {'MATCH' if len(cc) == EXPECTED_CLASSES else 'DIFFERENCE'}")
    add("  (A difference can mean: the mirror ships a SUBSET/different snapshot than the")
    add("   published dataset card, a partial download, or extra/missing folders.")
    add("   Mirror differences are documented in docs/dataset_research.md -")
    add("   investigate before training; do NOT silently assume 80/6,900 locally.)")
    add("")
    add("-" * 78)
    add("8. WARNINGS (automatic)")
    add("-" * 78)
    warnings = []
    if counts and min(counts) > 0 and max(counts) > 0:
        ratio = max(counts) / min(counts)
        if ratio > 3:
            warnings.append(f"class imbalance ratio max/min = {ratio:.1f} "
                            f"({max(counts)} vs {min(counts)}) - use stratified splits, "
                            f"class weights, macro-F1")
        small = {k: v for k, v in cc.items() if v < args.min_per_class}
        if small:
            listed = ", ".join(f"{k}({v})" for k, v in sorted(small.items(), key=lambda x: x[1]))
            warnings.append(f"{len(small)} class(es) below {args.min_per_class} images: "
                            f"{listed} - a stratified split may be impossible for them; "
                            f"consider augmentation or excluding them")
    if stats["loose_images"]:
        warnings.append(f"{stats['loose_images']} image(s) are not inside any class folder "
                        f"(unlabeled) - move or remove them before training")
    if stats["corrupt"]:
        warnings.append(f"{len(stats['corrupt'])} corrupt/unreadable file(s) - this script "
                        f"does NOT delete anything; remove them manually before training")
    if dup_groups:
        warnings.append(f"{len(dup_groups)} exact-duplicate group(s) - remove before "
                        f"splitting to avoid train/test leakage")
    if len(stats["size_counter"]) > 20:
        warnings.append(f"{len(stats['size_counter'])} distinct image sizes - "
                        f"the resizing step is mandatory before modeling")
    if not warnings:
        warnings.append("none triggered")
    for i, w in enumerate(warnings, 1):
        add(f"  {i}. {w}")
    add("")
    add("-" * 78)
    add("9. NEXT STEPS")
    add("-" * 78)
    add("  * Review warnings above; clean data into data/processed/ (future phase).")
    add("  * Read docs/project_pipeline.md for the planned train/val/test pipeline.")
    add("  * No model has been trained yet (assignment scope).")
    add("=" * 78)
    add("END OF REPORT")
    add("=" * 78)
    return "\n".join(L) + "\n"


def draw_chart(class_counts: dict, out_png: Path, title: str):
    """Horizontal bar chart of the class distribution. Returns (ok, message)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False, "matplotlib not installed -> pip install matplotlib"

    if not class_counts:
        return False, "no class folders to plot"
    items = sorted(class_counts.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig_h = max(6.0, 0.27 * len(items))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.barh(names, vals, color="#2e7d32")
    ax.set_xlabel("number of images")
    ax.set_title(title, fontsize=12)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    top = max(vals) if vals else 1
    for y, v in enumerate(vals):
        ax.text(v + top * 0.01, y, str(v), va="center", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return True, ""


# ----------------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="explore_dataset.py",
        description="Read-only inspection of data/raw -> reports/dataset_summary.txt "
                    "(+ class_distribution.png).",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA,
                   help=f"dataset folder to inspect (default: {DEFAULT_DATA})")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_REPORTS,
                   help=f"where to write reports (default: {DEFAULT_REPORTS})")
    p.add_argument("--skip-chart", action="store_true",
                   help="do not generate reports/class_distribution.png")
    p.add_argument("--fast", action="store_true",
                   help="skip full pixel decoding (faster, slightly weaker corruption check)")
    p.add_argument("--min-per-class", type=int, default=10,
                   help="warn for classes with fewer images than this (default: 10)")
    p.add_argument("--max-list", type=int, default=20,
                   help="max duplicate/corrupt entries listed in the report (default: 20)")
    return p


def no_data_exit(reason: str) -> int:
    say(f"[!] {reason}")
    say("")
    say("No report was written - this script never fabricates statistics.")
    say("Download the dataset first, then re-run:")
    say("    python scripts/download_dataset.py --source kaggle   # ~265 MB (Kaggle login)")
    say("    python scripts/download_dataset.py --source hf       # ~9.5 GB (no login)")
    say("    python scripts/download_dataset.py --source manual   # printed instructions")
    say("See data/README.md for details.")
    return 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir

    say("=" * 78)
    say(" explore_dataset.py - READ-ONLY dataset inspection")
    say("=" * 78)
    say(f"data dir : {data_dir}")
    say(f"reports  : {out_dir}")

    if not data_dir.is_dir():
        return no_data_exit(f"data directory not found: {data_dir}")

    root, classes, loose = find_dataset_root(data_dir)
    parquet_files = sorted(data_dir.rglob("*.parquet"))

    if not classes and not loose and not parquet_files:
        return no_data_exit(f"no images and no parquet shards found under {data_dir}")

    stats = None
    if classes or loose:
        if parquet_files:
            say(f"[note] {len(parquet_files)} parquet shard(s) also present - "
                f"inspecting the image folders instead.")
        if root is None:
            return no_data_exit("could not locate a class-folder structure")
        say(f"class-folder root detected: {root}")
        say(f"classes detected          : {len(classes)}")
        stats = inspect_folder(root, classes, loose, deep=not args.fast, base=data_dir)
    else:
        say(f"[note] parquet mode - {len(parquet_files)} shard(s) found (HF route).")
        stats = inspect_parquet(parquet_files, deep=not args.fast, base=data_dir)
        if stats is None:
            return no_data_exit("pyarrow missing - cannot inspect parquet shards")

    if stats["total_images"] == 0:
        return no_data_exit("0 readable images found (all files corrupt or unsupported)")

    # ---- write report ----
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "dataset_summary.txt"
    report_path.write_text(build_report(stats, args, data_dir), encoding="utf-8", newline="\n")
    say(f"\n[written] {report_path}")

    # ---- class distribution chart ----
    chart_path = out_dir / "class_distribution.png"
    if args.skip_chart:
        say("[skip] chart disabled via --skip-chart")
    else:
        ok, msg = draw_chart(stats["class_counts"], chart_path,
                             "Class distribution - medicinal leaf dataset")
        if ok:
            say(f"[written] {chart_path}")
        else:
            say(f"[skip] chart not created: {msg}")

    # ---- console summary ----
    cc = stats["class_counts"]
    counts = sorted(cc.values())
    dup_groups = sum(1 for v in stats["hash_paths"].values() if len(v) > 1)
    say("")
    say("-" * 78)
    say("SUMMARY")
    say("-" * 78)
    say(f"  valid images   : {stats['total_images']}")
    say(f"  classes        : {len(cc)}")
    if counts:
        say(f"  images/class   : min {min(counts)} | median {statistics.median(counts):.0f} | "
            f"max {max(counts)}")
    say(f"  corrupt files  : {len(stats['corrupt'])}")
    say(f"  duplicate grps : {dup_groups}")
    say(f"  total size     : {hbytes(stats['total_bytes'])}")
    say(f"  expected       : {EXPECTED_IMAGES} images / {EXPECTED_CLASSES} classes "
        f"(verified dataset card)")
    say("-" * 78)
    say("Done. Source data was NOT modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())





