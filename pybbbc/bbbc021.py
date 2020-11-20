import os
import re
import numpy as np
import pandas as pd
import zipfile
from datetime import datetime
import h5py
from skimage import io
from tqdm.auto import tqdm
from .utils import download_file, correct_illumination, scale_pixel_intensity

class BBBC021:

    IMG_SHAPE = (3, 1024, 1280)
    CHANNELS = ['Actin', 'Tubulin', 'DAPI']
    N_SITES = 4

    def __init__(self, path='data/bbbc021.h5' ,labeled=None, compound=None, moa=None):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    @staticmethod
    def make_dataset(file_path='data/bbbc021.h5', data_dir='data'):
        def get_metadata(raw_data_dir):
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
            images_dir = os.path.join(data_dir, 'raw', 'images')
            filename = [
                x
                for x in os.listdir(images_dir)
                if (x.endswith('.zip'))
                and ('Week4_27521' in x)
            ][0]
            file_path = os.path.join(images_dir, filename)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(images_dir)
            return os.path.join(images_dir, plate)
                
        def process_channel(channel, plate_df, plate_dir, plate_h5):
            n_images = len(plate_df)
            channel_imgs = np.empty((n_images,)+BBBC021.IMG_SHAPE[1:], dtype=np.float16)
            sites_tqdm = tqdm(total=BBBC021.N_SITES, desc='Sites', leave=False)
            for s in range(1, BBBC021.N_SITES+1):
                filenames = plate_df.loc[
                    plate_df.Image_Metadata_Site == s,
                    'Image_FileName_{}'.format(channel)
                ].tolist()
                
                filenames_tqdm = tqdm(total=len(filenames), desc='Reading images', leave=False)
                for i, filename in enumerate(filenames):
                    img = io.imread(os.path.join(plate_dir, filename)).astype(np.float16)
                    channel_imgs[(s-1)*len(filenames)+i] = img
                    filenames_tqdm.update(1)
                
                filenames_tqdm.set_description('Computing illumination correction')
                channel_imgs[(s-1)*len(filenames):(s)*len(filenames)] = \
                    correct_illumination(channel_imgs[(s-1)*len(filenames):(s)*len(filenames)])
                
                filenames_tqdm.close()
                sites_tqdm.update(1)

            sites_tqdm.set_description('Scaling pixel values')
            channel_imgs = scale_pixel_intensity(channel_imgs)
            
            sites_tqdm.set_description('Saving preprocessed images')
            with h5py.File(plate_h5, 'a') as h5_file:
                h5_file['images'][:, c, ...] = channel_imgs
                
            sites_tqdm.close()

        def make_virtual_dataset(vds_path, hdf5_dir, n_images):
            with h5py.File(os.path.join(hdf5_dir, os.listdir(hdf5_dir)[0]), 'r') as h5_file:
                datasets = {
                    x: [h5_file[x].shape, h5_file[x].dtype]
                    for x in list(h5_file.keys())
                }

            layouts = {
                x: h5py.VirtualLayout(shape=(n_images,) + shape[1:], dtype=dtype)
                for x, (shape, dtype) in datasets.items()
            }

            for i, filename in enumerate(os.listdir(hdf5_dir)):
                file_path = os.path.join(hdf5_dir, filename)
                for d, (s, _) in datasets.items():
                    layouts[d][i*s[0]:(i+1)*s[0]] = h5py.VirtualSource(file_path, d, s)
                    
            with h5py.File(vds_path, "w") as h5_file:
                h5_file.attrs['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                h5_file.attrs['h5py_info'] = h5py.version.info
                h5_file.attrs['dataset'] = 'bbbc021'
                h5_file.attrs['website'] = 'https://bbbc.broadinstitute.org/BBBC021'
                for name, layout in layouts.items():
                    h5_file.create_virtual_dataset(name, layout)
            
        raw_data_dir = os.path.join(data_dir, 'raw')
        hdf5_dir = os.path.join(data_dir, 'hdf5')
        if not os.path.exists(hdf5_dir):
            os.mkdir(hdf5_dir)

        metadata_df = get_metadata(raw_data_dir)
        plates = metadata_df.Image_Metadata_Plate_DAPI.unique().tolist()
        plates_tqdm = tqdm(total=len(plates), desc='Plates', leave=False)
        for plate in plates:
            plate_df = metadata_df.loc[
                metadata_df.Image_Metadata_Plate_DAPI == plate
            ]
            n_images = len(plate_df)

            channels_tqdm = tqdm(total=BBBC021.IMG_SHAPE[0], desc='Extracting images...', leave=False)
            plate_dir = extract_plate(plate)
            
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
            
            channels_tqdm.set_description('Channels')
            for c, channel in enumerate(BBBC021.CHANNELS):
                process_channel(channel, plate_df, plate_dir, h5_file_path)
                channels_tqdm.update(1)
            
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
            plates_tqdm.update(1)

        plates_tqdm.set_description('Finalizing')
        make_virtual_dataset(file_path, hdf5_dir, len(metadata_df))
        plates_tqdm.close()

    @staticmethod
    def download_raw_data(data_path=None):
        if data_path is None:
            data_path = os.getcwd()

        urls_images = 'https://github.com/giacomodeodato/pybbbc/raw/main/bbbc021/urls_images.txt'
        urls_metadata = 'https://github.com/giacomodeodato/pybbbc/raw/main/bbbc021/urls_metadata.txt'

        data_dir = os.path.join(data_path, 'data')
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        raw_data_dir = os.path.join(data_dir, 'raw')
        if not os.path.exists(raw_data_dir):
            os.mkdir(raw_data_dir)

        urls_path = download_file(urls_metadata, dst_dir=data_dir)
        with open(urls_path) as file_object:
            for url in file_object:
                download_file(url, raw_data_dir)

        images_dir = os.path.join(raw_data_dir, 'images')
        if not os.path.exists(images_dir):
            os.mkdir(images_dir)
        urls_path = download_file(urls_images, dst_dir=data_dir)
        with open(urls_path) as file_object:
            pbar = tqdm(total=len(file_object.readlines()))
            file_object.seek(0)
            for url in file_object:
                download_file(url, images_dir)
                pbar.update(1)
            pbar.close()

