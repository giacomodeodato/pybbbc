"""
Functions for ownloading and compiling the BBBC021 dataset
"""

import re
import shutil
import zipfile
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy as np
import pandas as pd
from skimage import io
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

import pybbbc.constants as constants

from .image import correct_illumination, scale_pixel_intensity
from .utils import download_file, get_paths


def download(root_path: Union[str, Path]):
    """Downloads raw images and metadata to root_path/data/raw.

    Args:
        data_path: parent folder of the data directory.
            If None, default is current working directory.
    """

    root_path = Path(root_path).expanduser()

    # TODO: Include these with the package install via setup.py
    urls_images = "https://raw.githubusercontent.com/zbarry/pybbbc/main/metadata/urls_images.txt"  # noqa: E501
    urls_metadata = "https://raw.githubusercontent.com/zbarry/pybbbc/main/metadata/urls_metadata.txt"  # noqa: E501

    paths = get_paths(root_path)

    # create data directories
    data_dir, raw_data_dir, images_dir = (
        paths["data"],
        paths["raw_data"],
        paths["images"],
    )

    images_dir.mkdir(exist_ok=True, parents=True)

    # download metadata files
    urls_path = download_file(urls_metadata, dst_dir=data_dir)

    with open(urls_path) as file_object:
        for url in file_object:
            download_file(url, raw_data_dir)

    # download images files
    urls_path = download_file(urls_images, dst_dir=data_dir)

    with open(urls_path) as file_object:
        pbar = tqdm(total=len(file_object.readlines()))
        file_object.seek(0)

        for url in file_object:
            download_file(url, images_dir)
            pbar.update(1)

        pbar.close()


def make_dataset(root_path: Union[str, Path], max_workers: int = 8):
    """Creates a virtual HDF5 dataset with preprocessed images and metadata.

    Data should be previously downloaded using BBBC021.download_raw_data().

    Args:
        data_path : str, optional
            Parent folder of the data directory.
            Default is `~/.cache/`
    """

    # data directories

    paths = get_paths(root_path)

    data_dir, raw_data_dir, hdf5_dir = (
        paths["data"],
        paths["raw_data"],
        paths["hdf5"],
    )

    hdf5_dir.mkdir(exist_ok=True, parents=True)

    # process metadata
    moa_df, image_df = load_metadata(raw_data_dir)

    metadata_df = get_metadata(moa_df, image_df)

    # # get plates and create progress bar
    # plates = metadata_df.Image_Metadata_Plate_DAPI.unique().tolist()

    # process_map(
    #     partial(
    #         process_plate,
    #         data_dir=data_dir,
    #         hdf5_dir=hdf5_dir,
    #         metadata_df=metadata_df,
    #     ),
    #     plates,
    #     max_workers=max_workers,
    # )

    # create virtual hdf5 dataset

    file_path = data_dir / "bbbc021.h5"

    make_virtual_dataset(file_path, hdf5_dir, len(metadata_df))


def load_metadata(raw_data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    moa_df = pd.read_csv(raw_data_dir / "BBBC021_v1_moa.csv")
    image_df = pd.read_csv(raw_data_dir / "BBBC021_v1_image.csv")

    return moa_df, image_df


def get_metadata(moa_df: pd.DataFrame, image_df: pd.DataFrame) -> pd.DataFrame:
    """Merges and preprocesses metadata files.

    Reads the image and moa metadata dataframes, creates the site
    column, merges the metadata and fills missing values with null.

    Returns : pandas.DataFrame
        The processed metadata DataFrame
    """

    image_df["Image_Metadata_Site"] = image_df.Image_FileName_DAPI.transform(
        lambda x: int(re.search("_s[1-4]_", x).group()[2])  # type: ignore
    )
    return (
        image_df.merge(
            moa_df,
            how="left",
            left_on=[
                "Image_Metadata_Compound",
                "Image_Metadata_Concentration",
            ],
            right_on=["compound", "concentration"],
        )
        .drop(columns=["compound", "concentration"])
        .fillna("null")
    )


def extract_plate(plate: str, data_dir: Path) -> Path:
    """Unzips the plate file.

    Args:
    plate : str
        Id of the plate to unzip.

    Returns : str
        The path to the extracted images.
    """

    images_dir = data_dir / "raw/images"

    zip_paths = list(images_dir.glob("*.zip"))

    try:
        file_path = [path for path in zip_paths if plate in path.name][0]
    except IndexError:
        print(f"Missing plate .zip file: {plate}")
        raise

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(images_dir)

    return images_dir / plate


def process_plate(
    plate: str, data_dir: Path, hdf5_dir: Path, metadata_df: pd.DataFrame
):
    """
    Extract the data from the plate .zip file and create corresponding hdf5
    """
    # get plate metadata
    plate_df = metadata_df.loc[metadata_df.Image_Metadata_Plate_DAPI == plate]
    n_images = len(plate_df)

    # create plate channels progress bar
    channels_tqdm = tqdm(
        total=constants.IMG_SHAPE[0], desc="Extracting images...", leave=False,
    )

    # extract plate
    plate_dir = extract_plate(plate, data_dir)

    # create plate hdf5 file
    channels_tqdm.set_description("Creating hdf5 dataset")

    h5_file_path = hdf5_dir / f"{plate}.h5"

    create_plate(h5_file_path, n_images)

    # process plate channels
    channels_tqdm.set_description("Channels")
    for channel_idx, channel in enumerate(constants.CHANNELS):
        process_channel(
            channel, channel_idx, plate_df, plate_dir, h5_file_path
        )
        channels_tqdm.update(1)

    # save metadata
    channels_tqdm.set_description("Saving metadata")
    save_metadata(h5_file_path, plate_df)
    channels_tqdm.close()

    # remove unzipped images
    shutil.rmtree(plate_dir, ignore_errors=True)


def create_plate(h5_file_path: Path, n_images: int):
    """
    Allocate space for the hdf5 arrays on disk for a given plate.
    """

    with h5py.File(h5_file_path, "w") as h5_file:
        h5_file.attrs["timestamp"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        h5_file.attrs["info"] = h5py.version.info
        h5_file.create_dataset(
            "images", (n_images,) + constants.IMG_SHAPE, np.float16
        )
        h5_file.create_dataset("site", (n_images,), np.uint8)
        h5_file.create_dataset(
            "well", (n_images,), h5py.string_dtype(encoding="utf-8")
        )
        h5_file.create_dataset("replicate", (n_images,), np.uint8)
        h5_file.create_dataset(
            "plate", (n_images,), h5py.string_dtype(encoding="utf-8")
        )
        h5_file.create_dataset(
            "compound", (n_images,), h5py.string_dtype(encoding="utf-8"),
        )
        h5_file.create_dataset("concentration", (n_images,), np.float16)
        h5_file.create_dataset(
            "moa", (n_images,), h5py.string_dtype(encoding="utf-8")
        )


def process_channel(channel, channel_idx, plate_df, plate_dir, plate_h5):
    """Processes the channel images of a plate.

    Computes and applies illumination correction per site.
    Scales pixel intensities.
    Saves the preprocessed images in the plate hdf5 file.

    Args:
    channel : int
        Channel of the plate to process.
    plate_df : pandas.DataFrame
        Metadata of the plate to process.
    plate_dir : str
        Path to the plate images directory.
    plate_h5 : str
        Path to the plate hdf5 file.
    """

    # create images array and sites progress bar
    n_images = len(plate_df)

    channel_imgs = np.empty(
        (n_images,) + constants.IMG_SHAPE[1:], dtype=np.float16
    )
    sites_tqdm = tqdm(total=constants.N_SITES, desc="Sites", leave=False)

    for s in range(1, constants.N_SITES + 1):

        # get filenames of images with site s
        filenames = plate_df.loc[
            plate_df.Image_Metadata_Site == s,
            "Image_FileName_{}".format(channel),
        ].tolist()

        # read images with site s
        filenames_tqdm = tqdm(
            total=len(filenames), desc="Reading images", leave=False
        )
        for i, filename in enumerate(filenames):
            img = io.imread(plate_dir / filename).astype(np.float16)
            channel_imgs[(s - 1) * len(filenames) + i] = img
            filenames_tqdm.update(1)

        # compute and apply illumination correction
        filenames_tqdm.set_description("Computing illumination correction")
        channel_imgs[
            (s - 1) * len(filenames) : (s) * len(filenames)
        ] = correct_illumination(
            channel_imgs[(s - 1) * len(filenames) : (s) * len(filenames)]
        )

        filenames_tqdm.close()
        sites_tqdm.update(1)

    # scale pixel intensities
    sites_tqdm.set_description("Scaling pixel values")
    channel_imgs = scale_pixel_intensity(channel_imgs)

    # save preprocessed images
    sites_tqdm.set_description("Saving preprocessed images")
    with h5py.File(plate_h5, "a") as h5_file:
        h5_file["images"][:, channel_idx, ...] = channel_imgs

    sites_tqdm.close()


def save_metadata(h5_file_path, plate_df: pd.DataFrame):
    with h5py.File(h5_file_path, "a") as h5_file:
        for s in range(1, 5):
            site_df = plate_df.loc[plate_df.Image_Metadata_Site == s]

            h5_file["site"][
                (s - 1) * len(site_df) : s * len(site_df)
            ] = site_df.Image_Metadata_Site.values.astype(np.uint8)

            h5_file["well"][
                (s - 1) * len(site_df) : s * len(site_df)
            ] = site_df.Image_Metadata_Well_DAPI.values

            h5_file["replicate"][
                (s - 1) * len(site_df) : s * len(site_df)
            ] = site_df.Replicate.values.astype(np.uint8)

            h5_file["plate"][
                (s - 1) * len(site_df) : s * len(site_df)
            ] = site_df.Image_Metadata_Plate_DAPI.values

            h5_file["compound"][
                (s - 1) * len(site_df) : s * len(site_df)
            ] = site_df.Image_Metadata_Compound.values

            h5_file["concentration"][
                (s - 1) * len(site_df) : s * len(site_df)
            ] = site_df.Image_Metadata_Concentration.values.astype(np.float16)

            h5_file["moa"][
                (s - 1) * len(site_df) : s * len(site_df)
            ] = site_df.moa.values


def make_virtual_dataset(vds_path, hdf5_dir, n_samples):
    """Creates virtual dataset merging the plates individual ones.

    Args:
    vds_path : str
        Path to the virtual dataset file.
    hdf5_dir : str
        Path to the hdf5 datasets directory.
    n_samples : int
        Number of samples in the final dataset.
    """

    # get datasets names and shapes
    with h5py.File(hdf5_dir / list(hdf5_dir.glob("*.h5"))[0]) as h5_file:
        datasets = {
            x: [h5_file[x].shape, h5_file[x].dtype]
            for x in list(h5_file.keys())
        }

    # create virtual layouts
    layouts = {
        x: h5py.VirtualLayout(shape=(n_samples,) + shape[1:], dtype=dtype)
        for x, (shape, dtype) in datasets.items()
    }

    # fill the virtual layouts
    for i, file_path in enumerate(list(hdf5_dir.glob("*.h5"))):
        for d, (s, _) in datasets.items():
            layouts[d][i * s[0] : (i + 1) * s[0]] = h5py.VirtualSource(
                file_path, d, s
            )

    # create virtual dataset
    with h5py.File(vds_path, "w") as h5_file:
        h5_file.attrs["timestamp"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        h5_file.attrs["h5py_info"] = h5py.version.info
        h5_file.attrs["dataset"] = "bbbc021"
        h5_file.attrs["github"] = "https://github.com/giacomodeodato/pybbbc"
        h5_file.attrs["website"] = "https://bbbc.broadinstitute.org/BBBC021"
        for name, layout in layouts.items():
            h5_file.create_virtual_dataset(name, layout)
