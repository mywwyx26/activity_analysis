import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile

def label_neuropils(neuropil_mask):
    """
    neuropil_mask: (y, x) with integer labels for each neuropil region (0 = background)
    Assigns names based on which label has a pixel closest to each corner:
        medulla       -> bottom right
        lobula        -> top right
        lobula plate  -> top left
    """
    y_size, x_size = neuropil_mask.shape
    labels = [l for l in np.unique(neuropil_mask) if l != 0]

    corners = {
        'medulla': (y_size - 1, x_size - 1),   # bottom right
        'lobula': (0, x_size - 1),              # top right
        'lobula plate': (0, 0),                 # top left
    }

    # all pixel coordinates for each label
    label_coords = {label: np.where(neuropil_mask == label) for label in labels}

    label_to_name = {}
    for name, (cy, cx) in corners.items():
        min_dists = {}
        for label, (ys, xs) in label_coords.items():
            dists = np.hypot(ys - cy, xs - cx)
            min_dists[label] = dists.min()
        closest_label = min(min_dists, key=min_dists.get)
        label_to_name[closest_label] = name

    return label_to_name


def neuropil_traces(data, neuropil_mask, label_to_name):
    traces = {}
    for label, name in label_to_name.items():
        region = neuropil_mask == label
        pixels = data[:, region]
        total = pixels.mean(axis=1)
        dff = (total - np.median(total)) / np.median(total)
        traces[name] = dff
    return traces


if __name__ == "__main__":
    input_folder = "C:\\Users\\megan\\flies\\activity_analysis"
    tif_videos = [f'{input_folder}\\registered\\{f}' for f in sorted(os.listdir(f'{input_folder}\\registered')) if f.endswith(".tif")]
    avg_images = [f'{input_folder}\\avg\\{f}' for f in sorted(os.listdir(f'{input_folder}\\avg')) if f.endswith(".tif")]
    clahe_images = [f'{input_folder}\\clahe\\{f}' for f in sorted(os.listdir(f'{input_folder}\\clahe')) if f.endswith(".npy")]
    neuropil_masks = [f'{input_folder}\\neuropils\\{f}' for f in sorted(os.listdir(f'{input_folder}\\neuropils')) if f.endswith(".tif")]

    fig1, ax1 = plt.subplots(4, 1, figsize=(16,8), sharex=True, sharey=True)
    fig2, ax2 = plt.subplots(4, 1, figsize=(16,8), sharex=True, sharey=True)
    count = 0
    for video_path, avg_path, clahe_path, neuropil_path in zip(tif_videos, avg_images, clahe_images, neuropil_masks):
        data = tifffile.imread(video_path)
        avg_data = tifffile.imread(avg_path)
        clahe_image = np.load(clahe_path)
        neuropils = tifffile.imread(neuropil_path)

        title = os.path.splitext(os.path.basename(video_path))[0]

        total = np.mean(data, axis=(1,2))
        total = (total - np.median(total)) / np.median(total)
        ax1[count].plot(total)
        ax1[count].set_title(title)

        label_to_name = label_neuropils(neuropils)
        traces = neuropil_traces(data, neuropils, label_to_name)
        for name, dff in traces.items():
            ax2[count].plot(dff, label=name)
        ax2[count].set_title(title)
        ax2[count].legend()

        count += 1

    fig1.tight_layout()
    fig1.savefig('total_activity.svg')

    fig2.tight_layout()
    fig2.savefig('neuropil_activity.svg')
    plt.show()