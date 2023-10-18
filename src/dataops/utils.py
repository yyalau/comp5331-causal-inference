from __future__ import annotations

import gdown

import tarfile
from typing import Optional
import zipfile

from pathlib import Path


def unzip(file_path: str) -> str:
    extract_dir = Path(file_path).parent.name

    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

    elif file_path.endswith(".tar"):
        with tarfile.open(file_path, "r:") as tar_ref:
            tar_ref.extractall(extract_dir)

    elif file_path.endswith(".tar.gz"):
        with tarfile.open(file_path, "r:gz") as tar_gz_ref:
            tar_gz_ref.extractall(extract_dir)

    return extract_dir


def download_from_gdrive(url: str, destination: str) -> str:
    destination_path = Path(destination)

    if not destination_path.exists():
        destination_path.parent.mkdir(parents=True, exist_ok=True)

    file_path: Optional[str] = gdown.download(url, destination, quiet=False)
    if file_path is None:
        raise IOError("Couldn't download data from Gdrive.")

    return file_path
