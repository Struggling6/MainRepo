from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


GITHUB_DOI_ZIP_URL = (
    "https://github.com/AlexisPapaioannou/Power-Consumption-Anomaly-Dataset/"
    "archive/refs/heads/main.zip"
)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download and extract the Power Consumption Anomaly Dataset into "
            "fl-backend/datasets/PowerConsumptionAnomaly."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fl-backend/datasets/PowerConsumptionAnomaly"),
        help="Directory where the extracted dataset should be placed.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=Path("fl-backend/datasets/PowerConsumptionAnomaly.zip"),
        help="Where to store the downloaded zip archive.",
    )
    parser.add_argument(
        "--source",
        choices=["github"],
        default="github",
        help="Download source. GitHub is easier for automation.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Extract an already downloaded archive_path instead of downloading.",
    )
    args = parser.parse_args()

    url = GITHUB_DOI_ZIP_URL

    if not args.skip_download:
        print(f"Downloading {url}")
        download_file(url, args.archive_path)

    print(f"Extracting {args.archive_path} to {args.output_dir}")
    extract_archive(args.archive_path, args.output_dir)

    csv_files = sorted(args.output_dir.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    for path in csv_files[:10]:
        print(f"  {path}")
    if len(csv_files) > 10:
        print("  ...")


if __name__ == "__main__":
    main()
