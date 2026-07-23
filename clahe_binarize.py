# apply clahe to all AVG images (only needs to be run once)
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from removebg import get_video_folders

input_folder = "C:\\Users\\megan\\flies\\activity_analysis"


video_folders = get_video_folders(input_folder)

fig, axes = plt.subplots(3, len(video_folders))
if len(video_folders) == 1:
    axes = axes.reshape(3, 1)  # keep 2D indexing consistent for a single video

for i, vf in enumerate(video_folders):
    avg_path = os.path.join(vf['folder'], f"AVG_{vf['base']}.tif")
    image = cv2.imread(avg_path, cv2.IMREAD_UNCHANGED)
    image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    axes[0, i].imshow(image, cmap='gray')
    axes[0, i].axis('off')

    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16, 16))
    clahe_image = clahe.apply(image_8bit)
    axes[1, i].imshow(clahe_image, cmap='gray')
    axes[1, i].axis('off')

    binarized = np.where(clahe_image > np.mean(clahe_image), 255, 0).astype(np.uint8)
    axes[2, i].imshow(binarized, cmap='gray')
    axes[2, i].axis('off')

    # save both outputs into the video's own subfolder, named after the AVG image
    avg_base = f"AVG_{vf['base']}"
    np.save(os.path.join(vf['folder'], f'{avg_base}_clahe.npy'), clahe_image)
    np.save(os.path.join(vf['folder'], f'{avg_base}_binarized.npy'), binarized)
    print(f"clahe/binarize done: {vf['folder']}")

plt.show()