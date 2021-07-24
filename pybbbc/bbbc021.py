"""
BBBC021 class definition for creating and working with BBBC021 dataset.
"""

from collections import namedtuple
from functools import cached_property
from pathlib import Path
from typing import Tuple, Union

import h5py
import janitor
import numpy as np
import pandas as pd
from numpy.lib.arraysetops import isin

from pybbbc import constants

from .dataset import download, make_dataset
from .utils import get_paths

Metadata = namedtuple("Metadata", ["plate", "compound"])
Plate = namedtuple("Plate", ["site", "well", "replicate", "plate"])
Compound = namedtuple("Compound", ["compound", "concentration", "moa"])


class BBBC021:
    """
    BBBC021 dataset class.

    Attributes:
        IMG_SHAPE : tuple
            Shape of the images: (C, H, W).
        CHANNELS : list
            Names of the channels.
        N_SITES : int
            Number of sites for each well.
        PLATES : list
            List of plate IDs.
        COMPOUNDS : list
            List of compounds.
        MOA : list
            List of Mechanisms of Action.

    Methods:
        make_dataset(data_path=None)
            Creates a virtual HDF5 dataset with preprocessed images and
            metadata.
        download_raw_data(data_path=None)
            Downloads raw images and metadata.
    """

    IMG_SHAPE = constants.IMG_SHAPE
    CHANNELS = constants.CHANNELS
    N_SITES = constants.N_SITES
    PLATES = constants.PLATES
    COMPOUNDS = constants.COMPOUNDS
    MOA = constants.MOA

    def __init__(self, root_path="~/.cache/", **kwargs):
        """Initializes the BBBC021 dataset.

        Args:
            path : str, optional
                Path to the virtual HDF5 dataset.
                Default is '~/.cache/bbbc021/bbbc021.h5'.

        Returns: instance of the BBBC021 dataset
        """

        root_path = Path(root_path).expanduser()

        self.root_path = root_path

        self._paths = get_paths(root_path)

        compiled_hdf5_path = self._paths["compiled_hdf5"]

        if not compiled_hdf5_path.exists():
            raise RuntimeError(
                "Dataset not found at '{}'.\n Use BBBC021.download() to "
                "download raw data and BBBC021.make_dataset() to preprocess "
                "and create the dataset.".format(compiled_hdf5_path)
            )

        self.dataset = h5py.File(compiled_hdf5_path, "r")
        self.index_vector = np.arange(
            self.dataset["moa"].shape[0]  # pylint: disable=no-member
        )

        # filter dataset based on kwargs query

        for k, v in kwargs.items():
            if not isinstance(v, (list, tuple, set)):
                v = [v]

            bool_vector = np.zeros_like(self.index_vector)

            for e in v:
                if isinstance(e, str):
                    e = bytes(e, "utf-8")

                bool_vector = bool_vector + np.array(
                    self.dataset[k][self.index_vector] == e
                )

            self.index_vector = self.index_vector[np.flatnonzero(bool_vector)]

    @cached_property
    def image_df(self) -> pd.DataFrame:
        def bytes_to_str(bts):
            return bts.decode("utf-8")

        return (
            pd.DataFrame(
                dict(
                    site=self.sites,
                    well=self.wells,
                    replicate=self.replicates,
                    plate=self.plates,
                    compound=self.compounds,
                    concentration=self.concentrations,
                    moa=self.moa,
                )
            )
            .transform_column("well", bytes_to_str)
            .transform_column("plate", bytes_to_str)
            .transform_column("compound", bytes_to_str)
            .transform_column("moa", bytes_to_str)
        ).astype(
            dict(
                well="category",
                plate="category",
                compound="category",
                moa="category",
            )
        )

    @cached_property
    def moa_df(self) -> pd.DataFrame:
        """
        Return a 3 column `DataFrame` with every combination of compound,
        concentration, and mechanism-of-action.
        Includes compounds with unknown MoA.
        """
        return (
            self.image_df[["compound", "concentration", "moa"]]
            .drop_duplicates()
            .sort_values(["compound", "concentration"])
            .reset_index(drop=True)
        )

    def metadata(self, index) -> Metadata:
        """
        Get metadata for compound at `index`.
        """

        row = self.image_df.iloc[index]

        site, well, replicate, plate, compound, concentration, moa = row[
            [
                "site",
                "well",
                "replicate",
                "plate",
                "compound",
                "concentration",
                "moa",
            ]
        ]

        metadata = Metadata(
            Plate(site, str(well), replicate, plate),
            Compound(compound, concentration, moa),
        )

        return metadata

    def __getitem__(self, index) -> Tuple[np.ndarray, Metadata]:

        index = self.index_vector[index]

        img = self.dataset["images"][index].astype(np.float32)

        metadata = self.metadata(index)

        return img, metadata

    def __len__(self):
        return len(self.index_vector)

    @property
    def images(self):
        """
        NOTE: This will load the entirety of the selected BBBC021 subset into
        memory.
        """
        return self.dataset["images"][self.index_vector]

    @cached_property
    def sites(self):
        return self.dataset["site"][self.index_vector]

    @cached_property
    def wells(self):
        return self.dataset["well"][self.index_vector]

    @cached_property
    def replicates(self):
        return self.dataset["replicate"][self.index_vector]

    @cached_property
    def plates(self):
        return self.dataset["plate"][self.index_vector]

    @cached_property
    def compounds(self):
        return self.dataset["compound"][self.index_vector]

    @cached_property
    def concentrations(self):
        return self.dataset["concentration"][self.index_vector]

    @cached_property
    def moa(self):
        return self.dataset["moa"][self.index_vector]

    @staticmethod
    def make_dataset(
        root_path: Union[str, Path] = "~/.cache/", max_workers: int = 4
    ):
        """Creates a virtual HDF5 dataset with preprocessed images and metadata.

        Data should be previously downloaded using BBBC021.download_raw_data().

        Args:
            data_path : str, optional
                Parent folder of the data directory.
                Default is `~/.cache/`
        """

        make_dataset(root_path, max_workers=max_workers)

    @staticmethod
    def download(root_path: Union[str, Path] = "~/.cache/"):
        """Downloads raw images and metadata to root_path/data/raw.

        Args:
            data_path: parent folder of the data directory.
                If None, default is current working directory.
        """
        download(root_path)
