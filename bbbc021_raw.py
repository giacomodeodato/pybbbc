import urllib.request
import os
from tqdm import tqdm

RAW_DATA_DIR='BBBC021/data/raw'

def gen_bar_updater(desc=None, leave=False):
    pbar = tqdm(total=None, leave=leave, desc=desc)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update

def download_file(url, dst_dir=None):
    filename = os.path.basename(url.strip())
    file_path = filename
    if dst_dir is not None:
        file_path = os.path.join(dst_dir, filename)
     
    urllib.request.urlretrieve(
        url, file_path,
        reporthook=gen_bar_updater(desc=filename)
    )
        
def download_raw_data(images_urls='bbbc021_images_urls.txt', metadata_urls='bbbc021_metadata_urls.txt'):
    if not os.path.exists(RAW_DATA_DIR):
        os.mkdir(RAW_DATA_DIR)
        
    with open(metadata_urls) as file_object:
        for url in file_object:
            download_file(url, RAW_DATA_DIR)
            
    images_dir = os.path.join(RAW_DATA_DIR, 'images')
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
        
    with open(images_urls) as file_object:
        for url in tqdm(file_object):
            download_file(url, images_dir)
            
download_raw_data()