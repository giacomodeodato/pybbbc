"""
BBBC021 class definition for creating and working with BBBC021 dataset.
"""

from pathlib import Path
from typing import Union

import h5py
import numpy as np

import pybbbc.constants as constants

from .dataset import download, make_dataset


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

    def __init__(self, path="~/.cache/bbbc021/bbbc021.h5", **kwargs):
        """Initializes the BBBC021 dataset.

        Args:
            path : str, optional
                Path to the virtual HDF5 dataset.
                Default is '~/.cache/bbbc021/bbbc021.h5'.

        Returns: instance of the BBBC021 dataset
        """

        path = Path(path).expanduser()

        if not path.exists():
            raise RuntimeError(
                "Dataset not found at '{}'.\n Use BBBC021.download() to "
                "download raw data and BBBC021.make_dataset() to preprocess "
                "and create the dataset.".format(path)
            )

        self.dataset = h5py.File(path, "r")
        self.index_vector = np.arange(self.dataset["moa"].shape[0])

        for k, v in kwargs.items():
            if (
                isinstance(v, list)
                or isinstance(v, tuple)
                or isinstance(v, set)
            ):
                bool_vector = np.zeros_like(self.index_vector)
                for e in v:
                    bool_vector = bool_vector + np.array(
                        self.dataset[k][self.index_vector] == e
                    )
                self.index_vector = self.index_vector[
                    np.nonzero(bool_vector)[0]
                ]
            else:
                self.index_vector = self.index_vector[
                    np.where(self.dataset[k][self.index_vector] == v)[0]
                ]

    def __getitem__(self, index):

        index = self.index_vector[index]

        img = self.dataset["images"][index].astype(np.float32)

        site = self.dataset["site"][index]
        well = self.dataset["well"][index].decode("utf-8")
        replicate = self.dataset["replicate"][index]
        plate = self.dataset["plate"][index].decode("utf-8")

        compound = self.dataset["compound"][index].decode("utf-8")
        concentration = self.dataset["concentration"][index]
        moa = self.dataset["moa"][index].decode("utf-8")

        metadata = (
            (site, str(well), replicate, plate),
            (compound, concentration, moa),
        )

        return img, metadata

    def __len__(self):
        return len(self.index_vector)

    @property
    def images(self):
        return self.dataset["images"][self.index_vector]

    @property
    def sites(self):
        return self.dataset["site"][self.index_vector]

    @property
    def wells(self):
        return self.dataset["well"][self.index_vector]

    @property
    def replicates(self):
        return self.dataset["replicate"][self.index_vector]

    @property
    def plates(self):
        return self.dataset["plate"][self.index_vector]

    @property
    def compounds(self):
        return self.dataset["compound"][self.index_vector]

    @property
    def concentrations(self):
        return self.dataset["concentration"][self.index_vector]

    @property
    def moa(self):
        return self.dataset["moa"][self.index_vector]

    @staticmethod
    def make_dataset(root_path: Union[str, Path] = "~/.cache/"):
        """Creates a virtual HDF5 dataset with preprocessed images and metadata.

        Data should be previously downloaded using BBBC021.download_raw_data().

        Args:
            data_path : str, optional
                Parent folder of the data directory.
                Default is `~/.cache/`
        """

        make_dataset(root_path)

    @staticmethod
    def download(root_path: Union[str, Path] = "~/.cache/"):
        """Downloads raw images and metadata to root_path/data/raw.

        Args:
            data_path: parent folder of the data directory.
                If None, default is current working directory.
        """
        download(root_path)
