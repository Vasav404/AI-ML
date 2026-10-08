#!/usr/bin/env python3
"""
download_dataset.py
===================
Phase 3 - download the SELECTED dataset:

    "Indian Medicinal Leaves Image Datasets" (paper name: DIMPSAR)
    Mendeley Data V3, DOI 10.17632/748f8jkphb.3, license CC BY 4.0
    80 classes / 6,900 leaf images (verified - see docs/dataset_research.md)

Routes
------
  --source kaggle   Kaggle API mirror (~265 MB). Needs a FREE Kaggle account and your
                    OWN credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY
                    environment variables). No keys are hardcoded or logged here.
  --source hf       HuggingFace mirror (~9.5 GB, 20 parquet shards). No account needed.
  --source manual   Prints step-by-step manual download instructions (Mendeley / Kaggle web).
  --source check    Tests connectivity + credential presence WITHOUT downloading anything.

Usage
-----
  python scripts/download_dataset.py --source check
  python scripts/download_dataset.py --source kaggle
  python scripts/download_dataset.py --source hf
  python scripts/download_dataset.py --source manual

Ethics / rules honored
----------------------
* Uses only official public APIs (Kaggle API v1, huggingface.co resolve URLs).
* Never scrapes websites, never bypasses logins, never hardcodes credentials.
* Downloaded data goes to data/raw/ and is never modified by this script.
* If a route needs manual work, this script TELLS YOU - it never fabricates data.

Exit codes: 0 = success, 1 = failure, 2 = usage error.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlsplit

# ----------------------------------------------------------------------------
# Configuration (public information only - no secrets)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = PROJECT_ROOT / "data" / "raw"

# Kaggle mirror of the selected dataset (verified 2026-09-30, CC BY 4.0)
KAGGLE_OWNER = "warcoder"
KAGGLE_SLUG = "indian-medicinal-plant-image-dataset"
KAGGLE_API_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_OWNER}/{KAGGLE_SLUG}"
)
KAGGLE_PAGE = f"https://www.kaggle.com/datasets/{KAGGLE_OWNER}/{KAGGLE_SLUG}"
KAGGLE_ZIP_NAME = "indian_medicinal_leaves_kaggle.zip"
KAGGLE_EXTRACT_DIR = "indian_medicinal_leaves_kaggle"

# HuggingFace mirror of the same dataset (public, no login; ~9.5 GB total)
HF_REPO = "Project-AgML/DIMPSAR_medicinal_leaf_classification"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/data"
HF_SHARDS = [f"train-{i:05d}-of-00020.parquet" for i in range(20)]
HF_DIR_NAME = "hf_parquet"

# Original source (manual route)
MANUAL_URL = "https://data.mendeley.com/datasets/748f8jkphb/3"
MANUAL_DOI = "10.17632/748f8jkphb.3"

# Expected size of the FULL dataset (verified - docs/dataset_research.md).
# Used only to print a friendly completeness hint, never as fabricated data.
EXPECTED_CLASSES = 80
EXPECTED_IMAGES = 6900

CHUNK = 1024 * 1024  # 1 MiB


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def say(msg: str = "", end: str = "\n") -> None:
    try:
        print(msg, flush=True, end=end)
    except OSError:
        pass  # stdout closed (e.g. piping) - never let logging kill the run


def load_kaggle_credentials():
    """Return (username, key) from env vars or ~/.kaggle/kaggle.json - or None.
    Credentials are never printed."""
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.is_file():
        try:
            data = json.loads(kaggle_json.read_text(encoding="utf-8"))
            user = data.get("username") or data.get("UserName")
            key = data.get("key") or data.get("Key")
            if user and key:
                return user, key
        except (OSError, ValueError) as exc:
            say(f"[warn] could not read kaggle.json: {exc.__class__.__name__}")
    return None


class _StripAuthRedirectHandler(HTTPRedirectHandler):
    """Follow redirects, but DROP the Authorization header when the host changes
    (Kaggle API 302 -> cloud storage) so credentials are never leaked."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        newreq = super().redirect_request(req, fp, code, msg, headers, newurl)
        if newreq is not None:
            old_host = urlsplit(req.full_url).netloc
            new_host = urlsplit(newurl).netloc
            if old_host and new_host and old_host != new_host:
                newreq.remove_header("Authorization")
        return newreq


def _open(url: str, auth_header: str | None = None, timeout: int = 90):
    headers = {"User-Agent": "medicinal-plant-classification/1.0 (college project)"}
    if auth_header:
        headers["Authorization"] = auth_header
    req = Request(url, headers=headers)
    opener = build_opener(_StripAuthRedirectHandler())
    return opener.open(req, timeout=timeout)


def _basic_auth(user: str, key: str) -> str:
    token = base64.b64encode(f"{user}:{key}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def stream_to_file(resp, dest: Path, label: str) -> int:
    """Download resp into dest with a simple progress display. Returns bytes written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = int(resp.headers.get("Content-Length") or 0)
    done = 0
    next_report = 5 * CHUNK
    with open(dest, "wb") as fh:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if done >= next_report:
                if total:
                    pct = 100.0 * done / total
                    say(f"\r  {label}: {done / 1e6:8.1f} MB / {total / 1e6:.1f} MB ({pct:5.1f}%)",
                        end="")
                else:
                    say(f"\r  {label}: {done / 1e6:8.1f} MB", end="")
                next_report = done + 5 * CHUNK
    if total:
        say(f"\r  {label}: {done / 1e6:8.1f} MB / {total / 1e6:.1f} MB (100.0%)")
    else:
        say(f"\r  {label}: {done / 1e6:8.1f} MB (done)")
    return done


def _count_images(folder: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def safe_extract(zip_path: Path, dest_dir: Path) -> int:
    """Extract with zip-slip protection. Returns number of extracted files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if not name or name.endswith("/"):
                continue
            target = (dest_dir / name).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise RuntimeError(f"blocked unsafe path in archive: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                while True:
                    chunk = src.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
            extracted += 1
    return extracted


# ----------------------------------------------------------------------------
# Route A: Kaggle API (credentials from the user, never hardcoded)
# ----------------------------------------------------------------------------
def download_kaggle(dest: Path, keep_zip: bool = False, force: bool = False) -> bool:
    say("\n=== Route: Kaggle API (mirror, CC BY 4.0, ~265 MB) ===")
    creds = load_kaggle_credentials()
    auth_header = None
    if creds:
        say(f"Kaggle credentials found for user '{creds[0]}' (key hidden).")
        auth_header = _basic_auth(*creds)
    else:
        say("No Kaggle credentials found - trying the public (unauthenticated) endpoint.")
        say("(If Kaggle rejects it: get a FREE token at kaggle.com -> Settings -> API ->")
        say(" 'Create New API Token', save kaggle.json to ~/.kaggle/, and re-run.)")

    extract_dir = dest / KAGGLE_EXTRACT_DIR
    if extract_dir.is_dir() and _count_images(extract_dir) > 0 and not force:
        say(f"Already extracted at {extract_dir} - nothing to do (use --force to redo).")
        return True

    zip_path = dest / KAGGLE_ZIP_NAME
    try:
        say(f"Requesting {KAGGLE_API_URL} ...")
        resp = _open(KAGGLE_API_URL, auth_header=auth_header)
        stream_to_file(resp, zip_path, "kaggle zip")
        resp.close()
    except HTTPError as exc:
        say(f"[error] Kaggle API returned HTTP {exc.code}.")
        if exc.code in (401, 403):
            say("       Kaggle requires credentials for this download now.")
            say("       Create a FREE API token (kaggle.com -> Settings -> API ->")
            say("       'Create New API Token') and put kaggle.json in ~/.kaggle/")
            say("       ... or use '--source hf' (no account needed).")
        elif exc.code == 404:
            say("       Dataset not found - the mirror may have moved; try '--source hf'.")
        if zip_path.exists():
            zip_path.unlink()
        return False
    except (URLError, OSError) as exc:
        say(f"[error] network problem: {exc}")
        return False

    try:
        n = safe_extract(zip_path, extract_dir)
        say(f"Extracted {n} files -> {extract_dir}")
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        say(f"[error] extraction failed: {exc}")
        return False
    finally:
        if zip_path.exists() and not keep_zip:
            zip_path.unlink()

    n_img = _count_images(extract_dir)
    say(f"Images now present: {n_img} (full dataset should be ~{EXPECTED_IMAGES} "
        f"images / {EXPECTED_CLASSES} classes).")
    say("Next: python scripts/explore_dataset.py")
    return n_img > 0


# ----------------------------------------------------------------------------
# Route B: HuggingFace (public, no account, LARGE ~9.5 GB)
# ----------------------------------------------------------------------------
def download_hf(dest: Path, force: bool = False) -> bool:
    say("\n=== Route: HuggingFace mirror (public, no login, ~9.5 GB total) ===")
    say("[note] This downloads 20 parquet shards - make sure you have disk space")
    say("       and do NOT run this inside a heavily synced folder if that is a problem.")
    out_dir = dest / HF_DIR_NAME
    ok = True
    for i, name in enumerate(HF_SHARDS, start=1):
        out_path = out_dir / name
        if out_path.exists() and out_path.stat().st_size > 0 and not force:
            say(f"  [{i:02d}/20] {name}: already present, skipping")
            continue
        url = f"{HF_BASE}/{name}"
        try:
            resp = _open(url)
            stream_to_file(resp, out_path, f"[{i:02d}/20] {name}")
            resp.close()
        except (HTTPError, URLError, OSError) as exc:
            say(f"\n  [error] failed on {name}: {exc}")
            ok = False
            break
    if ok:
        total = sum(p.stat().st_size for p in out_dir.glob('*.parquet'))
        say(f"Done. {len(list(out_dir.glob('*.parquet')))}/20 shards, {total / 1e9:.2f} GB.")
        say("Parquet shards need 'pyarrow' for inspection:  pip install pyarrow")
        say("Next: python scripts/explore_dataset.py")
    return ok


# ----------------------------------------------------------------------------
# Route C: manual download (printed instructions - we never pretend it happened)
# ----------------------------------------------------------------------------
def print_manual_instructions() -> None:
    say("""
=== Manual download (original source, no tools needed) ===

ROUTE 1 - Mendeley Data (ORIGINAL, recommended for citation)
  1. Open:  """ + MANUAL_URL + """
  2. Sign in / register (free) if Mendeley asks for it.
  3. Click 'Download All Files'.
  4. Unzip the download and place the folder(s) inside THIS project's  data/raw/  folder,
     so that class folders end up somewhere under data/raw/.
     Example final layout:
        data/raw/Indian Medicinal Leaves/<ClassName>/*.jpg

ROUTE 2 - Kaggle website (mirror, ~265 MB, no API needed)
  1. Open:  """ + KAGGLE_PAGE + """
  2. Click 'Download' (a free Kaggle login may be required).
  3. Unzip and place the contents under  data/raw/.

ROUTE 3 - HuggingFace (mirror, no login, ~9.5 GB parquet)
  1. Open:  https://huggingface.co/datasets/Project-AgML/DIMPSAR_medicinal_leaf_classification
  2. Download the files under  data/train-*-of-00020.parquet
     (or simply run:  python scripts/download_dataset.py --source hf)

AFTER ANY ROUTE
  python scripts/explore_dataset.py

LICENSE / CITATION (CC BY 4.0 - attribution is mandatory)
  B R, Pushpa; Rani, Shobha (2023), "Indian Medicinal Leaves Image Datasets",
  Mendeley Data, V3, doi: """ + MANUAL_DOI + """
""")


# ----------------------------------------------------------------------------
# Route: check (connectivity + credentials, downloads NOTHING)
# ----------------------------------------------------------------------------
def run_check(dest: Path) -> bool:
    say("=== Pre-flight check (no data is downloaded) ===")
    all_ok = True

    dest.mkdir(parents=True, exist_ok=True)
    say(f"[ok] data folder: {dest}")

    creds = load_kaggle_credentials()
    if creds:
        say("[ok] Kaggle credentials found (values hidden) -> '--source kaggle' ready")
    else:
        say("[--] Kaggle credentials NOT found -> '--source kaggle' would need setup")
        say("     (free: kaggle.com -> Account -> Create API Token -> kaggle.json)")

    try:
        req = Request(KAGGLE_API_URL, headers={"Range": "bytes=0-0",
                                               "User-Agent": "medicinal-plant-classification/1.0"})
        if creds:
            req.add_header("Authorization", _basic_auth(*creds))
        resp = build_opener(_StripAuthRedirectHandler()).open(req, timeout=30)
        status = getattr(resp, "status", 200)
        resp.close()
        if creds:
            say(f"[ok] Kaggle API reachable (HTTP {status}) -> ready to download")
        else:
            say(f"[ok] Kaggle public endpoint answered HTTP {status} -> unauthenticated "
                f"download appears open (credentials not required)")
    except HTTPError as exc:
        if creds:
            say(f"[fail] Kaggle API HTTP {exc.code} WITH your credentials")
            all_ok = False
        else:
            say(f"[--] Kaggle API answered HTTP {exc.code} (no credentials - expected)")
    except (URLError, OSError) as exc:
        say(f"[fail] Kaggle API unreachable: {exc}")
        all_ok = False

    hf_url = f"{HF_BASE}/{HF_SHARDS[0]}"
    try:
        req = Request(hf_url, headers={"Range": "bytes=0-0",
                                       "User-Agent": "medicinal-plant-classification/1.0"})
        resp = build_opener(_StripAuthRedirectHandler()).open(req, timeout=30)
        resp.close()
        say("[ok] HuggingFace reachable -> '--source hf' ready (no login needed)")
    except HTTPError as exc:
        say(f"[fail] HuggingFace HTTP {exc.code}")
        all_ok = False
    except (URLError, OSError) as exc:
        say(f"[fail] HuggingFace unreachable: {exc}")
        all_ok = False

    existing = _count_images(dest)
    parquets = len(list((dest / HF_DIR_NAME).glob("*.parquet")))
    if existing:
        say(f"[ok] found {existing} image file(s) under {dest} -> explore_dataset.py can run")
    elif parquets:
        say(f"[ok] found {parquets} parquet shard(s) under {dest} (pyarrow required to inspect)")
    else:
        say("[--] no dataset locally yet -> run '--source kaggle', '--source hf' or '--source manual'")

    say("\nCheck finished " + ("with no hard failures." if all_ok else "WITH FAILURES (see above)."))
    return all_ok


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_dataset.py",
        description="Download the Indian Medicinal Leaves (DIMPSAR) dataset into data/raw/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="License: CC BY 4.0 - cite: doi:" + MANUAL_DOI + " when using the data.",
    )
    p.add_argument("--source", choices=["kaggle", "hf", "manual", "check"],
                   help="which download route to use")
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                   help=f"destination folder (default: {DEFAULT_DEST})")
    p.add_argument("--keep-zip", action="store_true",
                   help="keep the downloaded zip instead of deleting it after extraction")
    p.add_argument("--force", action="store_true",
                   help="re-download even if data seems to exist already")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.source is None:
        parser.print_help()
        say("\n[!] no --source given - nothing was downloaded.")
        return 2

    dest = args.dest
    if args.source == "kaggle":
        return 0 if download_kaggle(dest, keep_zip=args.keep_zip, force=args.force) else 1
    if args.source == "hf":
        return 0 if download_hf(dest, force=args.force) else 1
    if args.source == "manual":
        print_manual_instructions()
        return 0
    return 0 if run_check(dest) else 1


if __name__ == "__main__":
    sys.exit(main())




