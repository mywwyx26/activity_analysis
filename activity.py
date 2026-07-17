import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile

def remove_background_pixels(data, avg_data, clahe_image):
    # gets rid of pixels lost to motion registration
    background_mask = np.any(data == 0, axis=0)  # shape: (y, x)
    data[:, background_mask] = 0
    avg_data[background_mask] = 0
    clahe_image[background_mask] = 0
    return background_mask


if __name__ == "__main__":
    input_folder = "C:\\Users\\megan\\flies\\activity_analysis"
    tif_videos = [f'{input_folder}\\registered\\{f}' for f in sorted(os.listdir(f'{input_folder}\\registered')) if f.endswith(".tif")]
    avg_images = [f'{input_folder}\\avg\\{f}' for f in sorted(os.listdir(f'{input_folder}\\avg')) if f.endswith(".tif")]
    clahe_images = [f'{input_folder}\\clahe\\{f}' for f in sorted(os.listdir(f'{input_folder}\\clahe')) if f.endswith(".npy")]

    fig1, ax1 = plt.subplots(4, 1, figsize=(16,8), sharex=True, sharey=True)
    count = 0
    for video_path, avg_path, clahe_path in zip(tif_videos, avg_images, clahe_images):
        data = tifffile.imread(video_path) # (t, y, x)
        avg_data = tifffile.imread(avg_path) # (y, x)
        clahe_image = np.load(clahe_path) # (y, x)

        mask = remove_background_pixels(data, avg_data, clahe_image) # edits data and avg_data and clahe_image
        foreground = data[:, ~mask] # (t, n_foreground_pixels)
        total = foreground.mean(axis=1) # (t) fluorescence trace over time
        total = (total - np.median(total)) / np.median(total) # deltaf/f
        
        ax1[count].plot(total)
        ax1[count].set_title(os.path.splitext(os.path.basename(video_path))[0])
        count += 1
    fig1.tight_layout()
    fig1.savefig('total_activity.svg')