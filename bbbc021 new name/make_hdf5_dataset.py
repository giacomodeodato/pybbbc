import os
import pandas as pd
import zipfile
from skimage import io
import numpy as np
from tqdm import tqdm
import shutil
import h5py
import re
from datetime import datetime

CHANNELS = ['Actin', 'Tubulin', 'DAPI']
IMG_SHAPE = (1024, 1280)
DATA_DIR='data'
RAW_DATA_DIR='data/raw'
raw_images_dir = os.path.join(RAW_DATA_DIR, 'images')
illum_dir = os.path.join(DATA_DIR, 'illumination_correction')
hdf5_dir = os.path.join(DATA_DIR, 'hdf5')
if not os.path.exists(hdf5_dir):
    os.mkdir(hdf5_dir)
    
image_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'BBBC021_v1_image.csv'))
image_df['Image_Metadata_Site'] = image_df.Image_FileName_DAPI.transform(
    lambda x: int(re.search('_s[1-4]_', x).group()[2])
)
moa_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'BBBC021_v1_moa.csv'))

plates = image_df.Image_Metadata_Plate_DAPI.unique().tolist()

str_dtype = h5py.string_dtype(encoding='utf-8')

for p, plate in enumerate(plates):
    zip_filename = [
        x
        for x in os.listdir(raw_images_dir)
        if plate in x
    ][0]
    plate_filename = os.path.splitext(zip_filename)[0]

    zip_path = os.path.join(raw_images_dir, zip_filename)
    plate_dir = os.path.join(raw_images_dir, plate_filename)
    
    print("[{:02d}/{}] Extracting images in {}... ".format(p, len(plates), plate_dir), end='')
    with zipfile.ZipFile(zip_path, 'r') as zipfile_object:
        zipfile_object.extractall(plate_dir)
    print("done.", end='\r')
    
    plate_df = image_df.loc[
        image_df.Image_Metadata_Plate_DAPI == plate
    ]
    plate_sub_dir = os.path.join(plate_dir, plate)
    illum_plate_dir = os.path.join(illum_dir, plate)
    
    h5_file_path = os.path.join(hdf5_dir, "{}.h5".format(plate))
    with h5py.File(h5_file_path, "w", ) as h5_file:
        
        h5_file.attrs['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        h5_file.attrs['info'] = h5py.version.info
        
        n_images = len(plate_df)
        images = h5_file.create_dataset(
            "images", 
            (n_images, len(CHANNELS)) + IMG_SHAPE,
             np.float16
        )
        site = h5_file.create_dataset(
            "site", 
            (n_images,),
             np.uint8
        )
        well = h5_file.create_dataset(
            "well", 
            (n_images,),
            str_dtype
        )
        replicate = h5_file.create_dataset(
            "replicate", 
            (n_images,),
            np.uint8
        )
        plate_id = h5_file.create_dataset(
            "plate", 
            (n_images,),
            str_dtype
        )
        compound = h5_file.create_dataset(
            "compound", 
            (n_images,),
            str_dtype
        )
        concentration = h5_file.create_dataset(
            "concentration", 
            (n_images,),
            np.float16
        )
        moa = h5_file.create_dataset(
            "moa", 
            (n_images,),
            str_dtype
        )
        
        for i, row in tqdm(
            plate_df.reset_index().iterrows(), 
            total=len(plate_df), 
            desc='[{:02d}/{}] {}'.format(p, len(plates), plate), 
            leave=False
        ):
            for c, channel in enumerate(CHANNELS):
                illum_name = '{}_s{}_illumination.npy'.format(channel, row.Image_Metadata_Site)
                illum_path = os.path.join(illum_plate_dir, illum_name)
                img_illum = np.load(illum_path)
                img_name = row['Image_FileName_{}'.format(channel)]
                img = io.imread(os.path.join(plate_sub_dir, img_name)).astype(np.float16)
                corr_img = img / img_illum
                
                images[i, c] = corr_img.astype(np.float16)
                site[i] = row.Image_Metadata_Site
                well[i] = row.Image_Metadata_Well_DAPI
                replicate[i] = row.Replicate
                plate_id[i] = row.Image_Metadata_Plate_DAPI
                compound[i] = row.Image_Metadata_Compound
                concentration[i] = row.Image_Metadata_Concentration
                moa_id = moa_df.loc[
                    (moa_df['compound'] == row.Image_Metadata_Compound) &\
                    (moa_df['concentration'] == row.Image_Metadata_Concentration), 
                    'moa'
                ]
                moa[i] = 'null' if moa_id.empty else moa_id.iloc[0]