import os
import pandas as pd
import zipfile
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import shutil

CHANNELS = ['DAPI', 'Tubulin', 'Actin']
DATA_DIR='data'
RAW_DATA_DIR='data/raw'
raw_images_dir = os.path.join(RAW_DATA_DIR, 'images')
illum_dir = os.path.join(DATA_DIR, 'illumination_correction')
if not os.path.exists(illum_dir):
    os.mkdir(illum_dir)

image_df = pd.read_csv('data/raw/BBBC021_v1_image.csv')
plates = image_df.Image_Metadata_Plate_DAPI.unique().tolist()

# For each plate
for plate in plates:
    zip_filename = [
        x 
        for x in os.listdir(raw_images_dir)
        if plate in x
    ][0]
    plate_filename = os.path.splitext(zip_filename)[0]
    
    zip_path = os.path.join(raw_images_dir, zip_filename)
    plate_dir = os.path.join(raw_images_dir, plate_filename)
    
    print("Extracting {} in {}... ".format(zip_path, plate_dir), end='')
    with zipfile.ZipFile(zip_path, 'r') as zipfile_object:
        zipfile_object.extractall(plate_dir)
    print("done.")
    plate_df = image_df.loc[
        image_df.Image_Metadata_Plate_DAPI == plate
    ]
    plate_sub_dir = os.path.join(plate_dir, plate)
    
    illum_plate_dir = os.path.join(illum_dir, plate)
    if not os.path.exists(illum_plate_dir):
        os.mkdir(illum_plate_dir)
        
    # For each channel
    for channel in CHANNELS:
        for site in range(1, 5):
            # Average images from all wells (exclude plate borders)
            img_avg = np.zeros((1024, 1280), dtype=np.float32)
            filenames = [
                x
                for x in plate_df['Image_FileName_{}'.format(channel)].tolist()
                if 's{}'.format(site) in x
            ]
            for filename in tqdm(filenames, desc='{} {} s{}'.format(plate, channel, site), leave=False):
                img = io.imread(os.path.join(plate_sub_dir, filename))
                img_avg = img_avg + img.astype(np.float32)
            img_avg = img_avg / len(filenames)
            
            # Apply Gaussian filter of size 500 and rescale
            img_mask = gaussian_filter(img_avg, sigma=500)
            robust_min = np.percentile(img_mask[img_mask > 0], 0.02)
            img_mask[img_mask < robust_min] = robust_min
            img_mask = img_mask / robust_min
            
            # and save resulting image
            img_mask_path = os.path.join(illum_plate_dir, '{}_s{}_illumination.npy'.format(channel, site))
            np.save(img_mask_path, img_mask.astype(np.float16))
            
    shutil.rmtree(plate_dir, ignore_errors=True)