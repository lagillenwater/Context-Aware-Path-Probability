"""Download Hetionet precomputed permutation graphs."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


DEFAULT_URL = (
    "https://github.com/hetio/hetionet/raw/"
    "a95ae76581af604e91d744680aee3f888fa18887/"
    "hetnet/permuted/matrix/hetionet-v1.0-permutations.zip"
)
DEFAULT_ZIP_FILENAME = "hetionet-v1.0-permutations.zip"
DEFAULT_EXTRACT_DIR_NAME = "hetionet-permutations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract Hetionet precomputed permutations."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Source URL for the permutations zip file.",
    )
    parser.add_argument(
        "--zip-filename",
        default=DEFAULT_ZIP_FILENAME,
        help="Filename to use under data/downloads/.",
    )
    parser.add_argument(
        "--extract-dir-name",
        default=DEFAULT_EXTRACT_DIR_NAME,
        help="Directory name under data/downloads/ for extracted content.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload zip even if it already exists.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract archive even if extracted directory exists.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Only download the zip file; skip extraction.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not args.url or not args.url.startswith(("http://", "https://")):
        raise ValueError("--url must start with http:// or https://")
    if not args.zip_filename:
        raise ValueError("--zip-filename cannot be empty")
    if not args.extract_dir_name:
        raise ValueError("--extract-dir-name cannot be empty")


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    data_dir = repo_root / "data"
    download_dir = data_dir / "downloads"
    zip_path = download_dir / args.zip_filename
    extract_dir = download_dir / args.extract_dir_name

    sys.path.insert(0, str(src_dir))
    from download_utils import download_file, extract_zip  # pylint: disable=import-error

    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository directory: {repo_root}")
    print(f"Download URL: {args.url}")
    print(f"Zip path: {zip_path}")

    if args.force_download or not zip_path.exists():
        download_file(args.url, zip_path)
    else:
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"Zip already exists ({size_mb:.1f} MB); skipping download.")

    if args.no_extract:
        print("Skipping extraction due to --no-extract.")
        return 0

    if args.force_extract and extract_dir.exists():
        print(f"Removing existing extracted directory: {extract_dir}")
        shutil.rmtree(extract_dir)

    if extract_dir.exists():
        print(f"Extracted directory already exists: {extract_dir}")
    else:
        extract_zip(zip_path, extract_dir)

    item_count = len(list(extract_dir.iterdir())) if extract_dir.exists() else 0
    print("Download Summary")
    print("=" * 40)
    print(f"Downloaded file: {zip_path.name}")
    print(f"Location: {zip_path}")
    print(f"Extracted directory: {extract_dir}")
    print(f"Extracted items: {item_count}")
    print("To use these permutations in analysis:")
    print("Set permutations_subdirectory = 'downloads/hetionet-permutations'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
