"""Implementation of file/directory operations."""

import gzip
import io
import json
import logging
import os
import pathlib
import shutil
import urllib.request
from functools import partial
from itertools import islice
from typing import Any, Dict

import gdown
import numpy as np
import yaml
from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)
LOGGER = logging.getLogger(__name__)


class _DownloadProgressBar(std_tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_file: str, filetype: str = ".txt.gz") -> None:
    """Download file from a given url.

    Args:
        url (str): `url` to the file that is downloaded
        output_file (str): Location where the downloaded file will be stored
        filetype (str, optional): Type of the file that is downloaded. Defaults to ".txt.gz".
    """
    with _DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)
        file, _ = urllib.request.urlretrieve(url, reporthook=t.update_to)

    if filetype == ".txt":
        os.rename(file, output_file)
    if filetype == ".txt.gz" or filetype == ".gz":
        batch_size: int = 1024**2  # 1MB batches
        # Approximate progress bar since unzipped file size is unknown.
        with tqdm(
            total=os.path.getsize(file) // batch_size * 1.5,
            desc="Unpacking binary file",
            unit="MB",
        ) as pbar:
            with open(output_file, "w", encoding="utf-8") as f_out:
                with gzip.open(file, "rb") as f_in:
                    while True:
                        file_content = f_in.read(batch_size).decode("utf-8", errors="ignore")
                        if not file_content:
                            break
                        f_out.write(file_content)
                        pbar.update(1)


def load_csv(file_path: str, num_of_lines: int = None):
    """Load comma separated value file.

    Args:
        file_path (str): path to the file.
        num_of_lines (int, optional): number of lines to load. Defaults to None.

    Returns:
        _type_: _description_
    """
    with io.open(file_path, "r", encoding="utf-8") as f:
        if num_of_lines is not None:
            data = np.loadtxt(islice(f, num_of_lines), delimiter=",")
        else:
            data = np.loadtxt(f, delimiter=",")
        return data


def move_files(source_folder: str, destination_folder: str) -> None:
    """Move the files.

    Args:
        source_folder (str): source path
        destination_folder (str): destination path
    """
    for file_name in os.listdir(source_folder):
        # construct full file path
        source = os.path.join(source_folder, file_name)
        destination = os.path.join(destination_folder, file_name)
        # move only files
        if os.path.isfile(source):
            shutil.move(source, destination)


def copy_files(source_path: pathlib.Path, destination_path: pathlib.Path) -> None:
    """Copy files from source folder to destination folder.

    Args:
        source_path (str): source path.
        destination_path (str): destination path.
    """
    if source_path.is_dir():
        for file_name in source_path.iterdir():
            # construct full file path
            source = source_path / file_name
            destination = destination_path / file_name
            # move only files
            if os.path.isfile(source):
                shutil.copy(source, destination)
    else:
        shutil.copyfile(source_path, destination_path)


def download_file_from_google_drive(file_id, image_root, filename):
    """Download a Google Drive file from  and place it in root.

    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
    """
    os.makedirs(image_root, exist_ok=True)
    gdown.download(id=file_id, output=os.path.join(image_root, filename), quiet=False)


def load_json(path: str) -> Dict[str, Any]:
    """Load json file to dictionary.

    Args:
        path (str): to the file

    Returns:
        Dict[str, Any]: dictionary from the json file
    """
    with open(path, "r") as file:
        data = json.load(file)
    return data


def save_json(path: str, data: dict):
    """Save dictionary to json.

    Args:
        path (str): to the file
        data (dict): dictonary to be saved
    """
    LOGGER.info("Saving dictionary to %s", path)
    path = pathlib.Path(path)
    root_dir: pathlib.Path = path.parent
    root_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def save_yaml(path: str, data: dict):
    """Save dictionary as YAML file.

    Args:
        path (str): to the file
        data (dict): to be saved
    """
    LOGGER.info("Saving dictionary to %s", path)
    with open(path, "w+") as f:
        yaml.dump(data, f)
