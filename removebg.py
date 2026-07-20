# remove pixels lost to motion registration (only needs to be run once)
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile

input_folder = "C:\\Users\\megan\\flies\\activity_analysis"
tif_videos = [f'{input_folder}\\registered\\{f}' for f in sorted(os.listdir(f'{input_folder}\\registered')) if f.endswith(".tif")]

for video_path in tif_videos:
    data = tifffile.imread(video_path) # (t, y, x)

    # get rid of pixels that were 0 for any point in time
    background_mask = np.any(data == 0, axis=0)  # shape: (y, x)
    data[:, background_mask] = 0
    tifffile.imwrite(video_path, data)
