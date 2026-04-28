#!/usr/bin/env python3
"""
Download model weights and (optionally) demo data from external hosting.

Supports Google Drive (public share links) and direct URLs.

Usage:
    # Download model weights only
    python scripts/download_assets.py --model

    # Download model + demo dataset
    python scripts/download_assets.py --model --demo

    # Download everything
    python scripts/download_assets.py --all

    # Use custom URLs (override defaults)
    python scripts/download_assets.py --model --model-url "https://drive.google.com/..."

Configuration:
    Edit DEFAULT_URLS below with your actual Google Drive share links.
    For Google Drive: use the sharing link (anyone with link can view).
    The script automatically converts sharing links to direct download URLs.
"""

import argparse
import hashlib
import os
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ================================================================
# EDIT THESE URLs WITH YOUR ACTUAL LINKS
# ================================================================
DEFAULT_URLS = {
    "model": {
        "url": "https://drive.google.com/file/d/1yAzNviPx5dDtTaQpsnA1vWKt4QLu9Uhy/view?usp=share_link",
        "dest": BASE_DIR / "model" / "ensemble_model.pt",
        "description": "Ensemble model weights (5-fold U-Net/ResNet-34)",
    },
    "demo_images": {
        "url": "https://drive.google.com/file/d/1AjWwzx1MIGaTDzl2DuugP-eKzZ0peM8T/view?usp=share_link",
        "dest": BASE_DIR / "demo_data" / "images.zip",
        "description": "Demo dataset images (Maricopa County patches)",
        "unzip_to": BASE_DIR / "demo_data",
    },
    "demo_meta": {
        "url": "https://drive.google.com/file/d/1yNXuHVJidp59b7VCyh0JKLNPJwP39pRQ/view?usp=share_link",
        "dest": BASE_DIR / "demo_data" / "metadata.csv",
        "description": "Demo dataset metadata CSV",
    },
}


def gdrive_direct_url(share_url: str) -> str:
    """Convert a Google Drive sharing link to a direct download URL."""
    # Handle /file/d/ID/view links
    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", share_url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    # Handle id=ID links
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", share_url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    # If it's already a direct URL, return as-is
    return share_url


def download_file(url: str, dest: Path, description: str = "") -> bool:
    """Download a file with progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        size_mb = dest.stat().st_size / 1e6
        print(f"  [SKIP] {dest.name} already exists ({size_mb:.1f} MB)")
        return True

    # Convert Google Drive links
    if "drive.google.com" in url:
        url = gdrive_direct_url(url)

    if "YOUR_GOOGLE_DRIVE_LINK" in url:
        print(f"  [!] URL not configured for {description}")
        print(f"      Edit DEFAULT_URLS in scripts/download_assets.py")
        print(f"      Or use --model-url / --demo-url flags")
        return False

    print(f"  Downloading: {description}")
    print(f"  URL: {url[:80]}...")
    print(f"  Destination: {dest}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "grid-dataset/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunks

            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        mb = downloaded / 1e6
                        print(f"\r  {mb:.1f} MB ({pct:.0f}%)", end="", flush=True)
                    else:
                        print(f"\r  {downloaded / 1e6:.1f} MB", end="", flush=True)
            print()

        size_mb = dest.stat().st_size / 1e6
        print(f"  [OK] Saved {dest.name} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  [FAIL] Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def unzip_file(zip_path: Path, dest_dir: Path):
    """Extract a zip file."""
    print(f"  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    print(f"  [OK] Extracted to {dest_dir}")


def main():
    p = argparse.ArgumentParser(
        description="Download model weights and demo data from external hosting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_assets.py --model
    python scripts/download_assets.py --all
    python scripts/download_assets.py --model --model-url "https://drive.google.com/file/d/XXX/view"
""",
    )
    p.add_argument("--model", action="store_true", help="Download model weights")
    p.add_argument("--demo", action="store_true", help="Download demo dataset")
    p.add_argument("--all", action="store_true", help="Download everything")
    p.add_argument("--model-url", default=None, help="Override model download URL")
    p.add_argument("--demo-url", default=None, help="Override demo data download URL")
    args = p.parse_args()

    if not (args.model or args.demo or args.all):
        p.print_help()
        print("\nSpecify --model, --demo, or --all")
        sys.exit(1)

    print("=" * 60)
    print("  Asset Downloader")
    print("=" * 60)

    success = True

    if args.model or args.all:
        url = args.model_url or DEFAULT_URLS["model"]["url"]
        ok = download_file(url, DEFAULT_URLS["model"]["dest"],
                           DEFAULT_URLS["model"]["description"])
        success = success and ok

    if args.demo or args.all:
        # Images (may be a zip)
        url = args.demo_url or DEFAULT_URLS["demo_images"]["url"]
        dest = DEFAULT_URLS["demo_images"]["dest"]
        ok = download_file(url, dest, DEFAULT_URLS["demo_images"]["description"])
        if ok and dest.suffix == ".zip":
            unzip_to = DEFAULT_URLS["demo_images"].get("unzip_to", dest.parent)
            unzip_file(dest, unzip_to)
        success = success and ok

        # Metadata
        meta_info = DEFAULT_URLS.get("demo_meta")
        if meta_info and "YOUR_" not in meta_info["url"]:
            ok = download_file(meta_info["url"], meta_info["dest"],
                               meta_info["description"])
            success = success and ok

    print()
    if success:
        print("[OK] All downloads complete.")
    else:
        print("[!] Some downloads failed. Check URLs and retry.")
    print("=" * 60)


if __name__ == "__main__":
    main()
