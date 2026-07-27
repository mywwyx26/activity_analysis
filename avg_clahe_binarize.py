# avg + clahe + binarize the registered video (only needs to be run once)
import os
import cv2
import numpy as np
import tifffile
from removebg import get_video_folders

input_folder = "C:\\Users\\megan\\flies\\activity_analysis"
video_folders = get_video_folders(input_folder)


for i, vf in enumerate(video_folders):
    video_path = os.path.join(vf['folder'], f"{vf['base']}.tif")
    data = tifffile.imread(video_path)  # (t, y, x)

    # avg
    avg_image = data.mean(axis=0)
    avg_path = os.path.join(vf['folder'], f"AVG_{vf['base']}.npy")
    np.save(avg_path, avg_image)
    avg_base = f"AVG_{vf['base']}"

    # clahe
    image_8bit = cv2.normalize(avg_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16, 16))
    clahe_image = clahe.apply(image_8bit)
    np.save(os.path.join(vf['folder'], f'{avg_base}_clahe.npy'), clahe_image)

    # binarize
    binarized = np.where(clahe_image > np.mean(clahe_image), 255, 0).astype(np.uint8)
    np.save(os.path.join(vf['folder'], f'{avg_base}_binarized.npy'), binarized)

    print(f"avg + clahe + binarize done: {vf['folder']}")