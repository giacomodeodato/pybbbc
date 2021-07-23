import os
import re
import numpy as np
import pandas as pd
import zipfile
from datetime import datetime
import h5py
import shutil
from skimage import io
from tqdm.auto import tqdm
from .utils import download_file, correct_illumination, scale_pixel_intensity
from pathlib import Path

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
            Creates a virtual HDF5 dataset with preprocessed images and metadata.
        download_raw_data(data_path=None)
            Downloads raw images and metadata.
    """

    IMG_SHAPE = (3, 1024, 1280)
    CHANNELS = ['Actin', 'Tubulin', 'DAPI']
    N_SITES = 4
    PLATES = [
        'Week10_40111', 'Week10_40115', 'Week10_40119', 'Week1_22123',
        'Week1_22141', 'Week1_22161', 'Week1_22361', 'Week1_22381',
        'Week1_22401', 'Week2_24121', 'Week2_24141', 'Week2_24161',
        'Week2_24361', 'Week2_24381', 'Week2_24401', 'Week3_25421',
        'Week3_25441', 'Week3_25461', 'Week3_25681', 'Week3_25701',
        'Week3_25721', 'Week4_27481', 'Week4_27521', 'Week4_27542',
        'Week4_27801', 'Week4_27821', 'Week4_27861', 'Week5_28901',
        'Week5_28921', 'Week5_28961', 'Week5_29301', 'Week5_29321',
        'Week5_29341', 'Week6_31641', 'Week6_31661', 'Week6_31681',
        'Week6_32061', 'Week6_32121', 'Week6_32161', 'Week7_34341',
        'Week7_34381', 'Week7_34641', 'Week7_34661', 'Week7_34681',
        'Week8_38203', 'Week8_38221', 'Week8_38241', 'Week8_38341',
        'Week8_38342', 'Week9_39206', 'Week9_39221', 'Week9_39222',
        'Week9_39282', 'Week9_39283', 'Week9_39301'
    ]
    COMPOUNDS = [
        "3,3'-diaminobenzidine", '5-fluorouracil', 'AG-1478', 'ALLN',
        'AZ-A', 'AZ-B', 'AZ-C', 'AZ-H', 'AZ-I', 'AZ-J', 'AZ-K', 'AZ-L',
        'AZ-M', 'AZ-N', 'AZ-O', 'AZ-U', 'AZ138', 'AZ235', 'AZ258', 'AZ701',
        'AZ841', 'Cdk1 inhibitor III', 'Cdk1/2 inhibitor (NU6102)', 'DMSO',
        'H-7', 'ICI-182,780', 'LY-294002', 'MG-132', 'PD-150606',
        'PD-169316', 'PD-98059', 'PP-2', 'SB-202190', 'SB-203580',
        'SP-600125', 'TKK', 'UNKNOWN', 'UO-126', 'Y-27632', 'acyclovir',
        'aloisine A', 'alsterpaullone', 'anisomycin', 'aphidicolin',
        'arabinofuranosylcytosine', 'atropine', 'bleomycin', 'bohemine',
        'brefeldin A', 'bryostatin', 'calpain inhibitor 2 (ALLM)',
        'calpeptin', 'camptothecin', 'carboplatin',
        'caspase inhibitor 1 (ZVAD)', 'cathepsin inhibitor I',
        'chlorambucil', 'chloramphenicol', 'cisplatin', 'colchicine',
        'cyclohexamide', 'cyclophosphamide', 'cytochalasin B',
        'cytochalasin D', 'demecolcine', 'deoxymannojirimycin',
        'deoxynojirimycin', 'docetaxel', 'doxorubicin', 'emetine',
        'epothilone B', 'etoposide', 'filipin', 'floxuridine', 'forskolin',
        'genistein', 'herbimycin A', 'hydroxyurea', 'indirubin monoxime',
        'jasplakinolide', 'lactacystin', 'latrunculin B', 'leupeptin',
        'methotrexate', 'methoxylamine', 'mevinolin/lovastatin',
        'mitomycin C', 'mitoxantrone', 'monastrol', 'neomycin',
        'nocodazole', 'nystatin', 'okadaic acid', 'olomoucine',
        'podophyllotoxin', 'proteasome inhibitor I', 'puromycin',
        'quercetin', 'raloxifene', 'rapamycin', 'roscovitine',
        'simvastatin', 'sodium butyrate', 'sodium fluoride',
        'staurosporine', 'taurocholate', 'taxol', 'temozolomide',
        'trichostatin', 'tunicamycin', 'valproic acid', 'vinblastine',
        'vincristine'
    ]
    MOA = [
        'Actin disruptors', 'Aurora kinase inhibitors',
        'Cholesterol-lowering', 'DMSO', 'DNA damage', 'DNA replication',
        'Eg5 inhibitors', 'Epithelial', 'Kinase inhibitors',
        'Microtubule destabilizers', 'Microtubule stabilizers',
        'Protein degradation', 'Protein synthesis', 'null'
    ]

    def __init__(self, path='~/.cache/bbbc021/bbbc021.h5', **kwargs):
        """Initializes the BBBC021 dataset.

        Args:
            path : str, optional
                Path to the virtual HDF5 dataset.
                Default is '~/.cache/bbbc021/bbbc021.h5'.

        Returns: instance of the BBBC021 dataset
        """

        path = Path(path).expanduser()

        if not path.exists():
            raise RuntimeError("Dataset not found at '{}'.\n Use BBBC021.download() to download raw data and BBBC021.make_dataset() to preprocess and create the dataset.".format(path))

        self.dataset = h5py.File(path, 'r')
        self.index_vector = np.arange(self.dataset['moa'].shape[0])

        for k, v in kwargs.items():
            if isinstance(v, list) or isinstance(v, tuple) or isinstance(v, set):
                bool_vector = np.zeros_like(self.index_vector)
                for e in v:
                    bool_vector = bool_vector + np.array(self.dataset[k][self.index_vector] == e)
                self.index_vector = self.index_vector[np.nonzero(bool_vector)[0]]
            else:
                self.index_vector = self.index_vector[np.where(self.dataset[k][self.index_vector] == v)[0]]

    def __getitem__(self, index):

        index = self.index_vector[index]

        img = self.dataset['images'][index].astype(np.float32)
        site = self.dataset['site'][index]
        well = self.dataset['well'][index]
        replicate = self.dataset['replicate'][index]
        plate = self.dataset['plate'][index]
        compound = self.dataset['compound'][index]
        concentration = self.dataset['concentration'][index]
        moa = self.dataset['moa'][index]

        metadata = (
            (
                site,
                well,
                replicate,
                plate
            ), (
                compound,
                concentration,
                moa
            )
        )

        return img, metadata

    def __len__(self):
        return len(self.index_vector)

    @property
    def images(self):
        return self.dataset['images'][self.index_vector]

    @property
    def sites(self):
        return self.dataset['site'][self.index_vector]

    @property
    def wells(self):
        return self.dataset['well'][self.index_vector]

    @property
    def replicates(self):
        return self.dataset['replicate'][self.index_vector]

    @property
    def plates(self):
        return self.dataset['plate'][self.index_vector]

    @property
    def compounds(self):
        return self.dataset['compound'][self.index_vector]

    @property
    def concentrations(self):
        return self.dataset['concentration'][self.index_vector]

    @property
    def moa(self):
        return self.dataset['moa'][self.index_vector]

    @staticmethod
    def make_dataset(data_path=None):
        """Creates a virtual HDF5 dataset with preprocessed images and metadata.

        Data should be previously downloaded using BBBC021.download_raw_data().

        Args:
            data_path : str, optional
                Parent folder of the data directory.
                Default is current working directory.
        """

        def get_metadata():
            """Merges and preprocesses metadata files.

            Reads the image and moa metadata dataframes, creates the site
            column, merges the metadata and fills missing values with null.

            Returns : pandas.DataFrame
                The processed metadata DataFrame
            """

            moa_df = pd.read_csv(os.path.join(raw_data_dir, 'BBBC021_v1_moa.csv'))
            image_df = pd.read_csv(os.path.join(raw_data_dir, 'BBBC021_v1_image.csv'))
            image_df['Image_Metadata_Site'] = image_df.Image_FileName_DAPI.transform(
                lambda x: int(re.search('_s[1-4]_', x).group()[2])
            )
            return image_df.merge(
                moa_df,
                how='left',
                left_on=['Image_Metadata_Compound', 'Image_Metadata_Concentration'],
                right_on=['compound', 'concentration']
            ).drop(
                columns=['compound', 'concentration']
            ).fillna('null')

        def extract_plate(plate):
            """Unzips the pate file.

            Args:
            plate : str
                Id of the plate to unzip.

            Returns : str
                The path to the extracted images.
            """

            images_dir = os.path.join(data_dir, 'raw', 'images')
            filename = [
                x
                for x in os.listdir(images_dir)
                if (x.endswith('.zip'))
                and (plate in x)
            ][0]
            file_path = os.path.join(images_dir, filename)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(images_dir)
            return os.path.join(images_dir, plate)

        def process_channel(channel, plate_df, plate_dir, plate_h5):
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
            channel_imgs = np.empty((n_images,)+BBBC021.IMG_SHAPE[1:], dtype=np.float16)
            sites_tqdm = tqdm(total=BBBC021.N_SITES, desc='Sites', leave=False)

            for s in range(1, BBBC021.N_SITES+1):

                # get filenames of images with site s
                filenames = plate_df.loc[
                    plate_df.Image_Metadata_Site == s,
                    'Image_FileName_{}'.format(channel)
                ].tolist()

                # read images with site s
                filenames_tqdm = tqdm(total=len(filenames), desc='Reading images', leave=False)
                for i, filename in enumerate(filenames):
                    img = io.imread(os.path.join(plate_dir, filename)).astype(np.float16)
                    channel_imgs[(s-1)*len(filenames)+i] = img
                    filenames_tqdm.update(1)

                # compute and apply illumination correction
                filenames_tqdm.set_description('Computing illumination correction')
                channel_imgs[(s-1)*len(filenames):(s)*len(filenames)] = \
                    correct_illumination(channel_imgs[(s-1)*len(filenames):(s)*len(filenames)])

                filenames_tqdm.close()
                sites_tqdm.update(1)

            # scale pixel intensities
            sites_tqdm.set_description('Scaling pixel values')
            channel_imgs = scale_pixel_intensity(channel_imgs)

            # save preprocessed images
            sites_tqdm.set_description('Saving preprocessed images')
            with h5py.File(plate_h5, 'a') as h5_file:
                h5_file['images'][:, c, ...] = channel_imgs

            sites_tqdm.close()

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
            with h5py.File(os.path.join(hdf5_dir, os.listdir(hdf5_dir)[0]), 'r') as h5_file:
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
            for i, filename in enumerate(os.listdir(hdf5_dir)):
                file_path = os.path.join(hdf5_dir, filename)
                for d, (s, _) in datasets.items():
                    layouts[d][i*s[0]:(i+1)*s[0]] = h5py.VirtualSource(file_path, d, s)

            # create virtual dataset
            with h5py.File(vds_path, "w") as h5_file:
                h5_file.attrs['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                h5_file.attrs['h5py_info'] = h5py.version.info
                h5_file.attrs['dataset'] = 'bbbc021'
                h5_file.attrs['github'] = 'https://github.com/giacomodeodato/pybbbc'
                h5_file.attrs['website'] = 'https://bbbc.broadinstitute.org/BBBC021'
                for name, layout in layouts.items():
                    h5_file.create_virtual_dataset(name, layout)

        if data_path is None:
            data_path = os.getcwd()

        # create data directories
        data_dir = os.path.join(data_path, 'data')
        raw_data_dir = os.path.join(data_dir, 'raw')
        hdf5_dir = os.path.join(data_dir, 'hdf5')
        if not os.path.exists(hdf5_dir):
            os.mkdir(hdf5_dir)

        # process metadata
        metadata_df = get_metadata()

        # get plates and create progress bar
        plates = metadata_df.Image_Metadata_Plate_DAPI.unique().tolist()
        plates_tqdm = tqdm(total=len(plates), desc='Plates', leave=False)

        for plate in plates:

            # get plate metadata
            plate_df = metadata_df.loc[
                metadata_df.Image_Metadata_Plate_DAPI == plate
            ]
            n_images = len(plate_df)

            # create plate channels progress bar
            channels_tqdm = tqdm(total=BBBC021.IMG_SHAPE[0], desc='Extracting images...', leave=False)

            # extract plate
            plate_dir = extract_plate(plate)

            # create plate hdf5 file
            channels_tqdm.set_description('Creating hdf5 dataset')
            h5_file_path = os.path.join(hdf5_dir, "{}.h5".format(plate))
            with h5py.File(h5_file_path, "w") as h5_file:
                h5_file.attrs['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                h5_file.attrs['info'] = h5py.version.info
                h5_file.create_dataset(
                    "images",
                    (n_images,) + BBBC021.IMG_SHAPE,
                    np.float16
                )
                h5_file.create_dataset("site", (n_images,), np.uint8)
                h5_file.create_dataset("well", (n_images,), h5py.string_dtype(encoding='utf-8'))
                h5_file.create_dataset("replicate", (n_images,), np.uint8)
                h5_file.create_dataset("plate", (n_images,), h5py.string_dtype(encoding='utf-8'))
                h5_file.create_dataset("compound", (n_images,), h5py.string_dtype(encoding='utf-8'))
                h5_file.create_dataset("concentration", (n_images,), np.float16)
                h5_file.create_dataset("moa", (n_images,), h5py.string_dtype(encoding='utf-8'))

            # process plate channels
            channels_tqdm.set_description('Channels')
            for c, channel in enumerate(BBBC021.CHANNELS):
                process_channel(channel, plate_df, plate_dir, h5_file_path)
                channels_tqdm.update(1)

            # save metadata
            channels_tqdm.set_description('Saving metadata')
            with h5py.File(h5_file_path, 'a') as h5_file:
                for s in range(1, 5):
                    site_df = plate_df.loc[
                        plate_df.Image_Metadata_Site == s
                    ]
                    h5_file['site'][(s-1)*len(site_df):s*len(site_df)] = site_df.Image_Metadata_Site.values.astype(np.uint8)
                    h5_file['well'][(s-1)*len(site_df):s*len(site_df)] = site_df.Image_Metadata_Well_DAPI.values
                    h5_file['replicate'][(s-1)*len(site_df):s*len(site_df)] = site_df.Replicate.values.astype(np.uint8)
                    h5_file['plate'][(s-1)*len(site_df):s*len(site_df)] = site_df.Image_Metadata_Plate_DAPI.values
                    h5_file['compound'][(s-1)*len(site_df):s*len(site_df)] = site_df.Image_Metadata_Compound.values
                    h5_file['concentration'][(s-1)*len(site_df):s*len(site_df)] = site_df.Image_Metadata_Concentration.values.astype(np.float16)
                    h5_file['moa'][(s-1)*len(site_df):s*len(site_df)] = site_df.moa.values
            channels_tqdm.close()

            # remove unzipped images
            plates_tqdm.set_description('Cleaning')
            shutil.rmtree(plate_dir, ignore_errors=True)
            plates_tqdm.update(1)

        # create virtual hdf5 dataset
        plates_tqdm.set_description('Finalizing')
        file_path = os.path.join(data_dir, 'bbbc021.h5')
        make_virtual_dataset(file_path, hdf5_dir, len(metadata_df))
        plates_tqdm.close()

    @staticmethod
    def download(data_path=None):
        """Downloads raw images and metadata to data_path/data/raw.

        Args:
            data_path: parent folder of the data directory.
                If None, default is current working directory.
        """

        if data_path is None:
            data_path = os.getcwd()

        # TODO: Include these with the package install via setup.py
        urls_images = 'https://raw.githubusercontent.com/giacomodeodato/pybbbc/main/metadata/urls_images.txt'
        urls_metadata = 'https://raw.githubusercontent.com/giacomodeodato/pybbbc/main/metadata/urls_metadata.txt'

        # create data directories
        data_dir = os.path.join(data_path, 'data')
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        raw_data_dir = os.path.join(data_dir, 'raw')
        if not os.path.exists(raw_data_dir):
            os.mkdir(raw_data_dir)
        images_dir = os.path.join(raw_data_dir, 'images')
        if not os.path.exists(images_dir):
            os.mkdir(images_dir)

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