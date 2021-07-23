"""
Helper functions for working with paths, downloading files, etc.
"""

import os
from pathlib import Path
from typing import Dict, Union
from urllib.error import HTTPError
from urllib.request import urlretrieve

from tqdm.auto import tqdm


def gen_bar_updater(desc=None, leave=False):
    pbar = tqdm(total=None, leave=leave, desc=desc)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def get_paths(root_dir: Union[str, Path]) -> Dict[str, Path]:
    root_dir = Path(root_dir).expanduser()

    data_dir = root_dir / "bbbc021"

    raw_data_dir = data_dir / "raw"

    images_dir = raw_data_dir / "images"

    hdf5_dir = data_dir / "hdf5"

    return dict(
        root=root_dir,
        data=data_dir,
        raw_data=raw_data_dir,
        hdf5=hdf5_dir,
        images=images_dir,
    )


def download_file(url: str, dst_dir: Union[Path, str] = "."):

    dst_dir = Path(dst_dir)

    filename = os.path.basename(url.strip())

    pbar = tqdm(total=None, leave=False, desc=filename)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    file_path = dst_dir / filename

    if file_path.exists():
        pbar.close()
        return file_path

    try:
        urlretrieve(url, file_path, reporthook=bar_update)
    except HTTPError:
        if file_path.exists():
            file_path.unlink()
        raise
    finally:
        pbar.close()

    return file_path
