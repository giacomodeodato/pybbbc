import os
import numpy as np
import urllib.request
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm

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
    pbar = tqdm(total=None, leave=False, desc=filename)
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    file_path = filename
    if dst_dir is not None:
        file_path = os.path.join(dst_dir, filename)
    if os.path.exists(file_path):
        pbar.close()
    else:
        try:
            urllib.request.urlretrieve(
                url, file_path,
                reporthook=bar_update
            )
        except:
            if os.path.exists(file_path):
                os.remove(file_path)
        finally:
            pbar.close()
    return file_path

def correct_illumination(images, sigma=500, min_percentile=0.02):
    img_avg = images.mean(axis=0)
    img_mask = gaussian_filter(img_avg.astype(np.float32), sigma=sigma).astype(np.float16)
    robust_min = np.percentile(img_mask[img_mask > 0], min_percentile)
    img_mask[img_mask < robust_min] = robust_min
    img_mask = img_mask / robust_min
    return images / img_mask

def scale_pixel_intensity(images):
    low = np.percentile(images, 0.1)
    high = np.percentile(images, 99.9)
    images = (images - low) / (high - low)
    return np.clip(images, 0, 1)