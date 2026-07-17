import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile

if __name__ == "__main__":
    input_folder = "C:\\Users\\megan\\flies\\activity_analysis"
    tif_videos = [f'{input_folder}\\registered\\{f}' for f in sorted(os.listdir(f'{input_folder}\\registered')) if f.endswith(".tif")]
    avg_images = [f'{input_folder}\\avg\\{f}' for f in sorted(os.listdir(f'{input_folder}\\avg')) if f.endswith(".tif")]
    clahe_images = [f'{input_folder}\\clahe\\{f}' for f in sorted(os.listdir(f'{input_folder}\\clahe')) if f.endswith(".npy")]

    for video_path, avg_path in zip(tif_videos, avg_images):
        data = tifffile.imread(video_path) # (t, y, x)
        avg_data = tifffile.imread(avg_path) # (y, x)

        total = np.mean(data, axis=(1,2)) # 1D array: fluorescence trace over time
        plt.plot(total)
    plt.show()