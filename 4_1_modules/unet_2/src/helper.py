import rasterio
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_window(tif_file_path, window):
    try:
        with rasterio.open(tif_file_path) as src:
            block_array = src.read(1, window=window)
            return np.unique(block_array)
    except Exception as e:
        print(f"Error processing window {window}: {e}")
        return np.array([])

def get_unique_values(tif_file_path, window_size=4096, max_workers=None):
    unique_values = set()
    windows = []

    with rasterio.Env(GDAL_CACHEMAX=1024):  # Increase cache size to 1024 MB
        with rasterio.open(tif_file_path) as src:
            height, width = src.height, src.width
            for i in range(0, height, window_size):
                for j in range(0, width, window_size):
                    window = rasterio.windows.Window(j, i, window_size, window_size)
                    windows.append(window)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_window, [tif_file_path]*len(windows), windows), total=len(windows)))

    for result in results:
        unique_values.update(result)

    return np.array(list(unique_values))