import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from removebg import get_video_folders

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


def grid_traces(data, n=4):
    """
    Splits data (t, y, x) into an n x n grid and computes a dF/F trace
    for each cell. Returns dict {(row, col): trace}.
    """
    t, y, x = data.shape
    y_edges = np.array_split(np.arange(y), n)
    x_edges = np.array_split(np.arange(x), n)

    traces = {}
    for i, ys in enumerate(y_edges):
        for j, xs in enumerate(x_edges):
            block = data[:, ys[:, None], xs]        # (t, len(ys), len(xs))
            total = block.mean(axis=(1, 2))          # (t,)
            dff = (total - np.median(total)) / np.median(total)
            traces[(i, j)] = dff
    return traces


def quadrant_plot(fig_title, cell_traces, n=4):
    """
    Groups the n x n grid cells into 4 quadrants (each made of a 2x2 block
    of grid cells) and plots each quadrant's 4 lines in its own subplot.
    """
    half = n // 2
    quadrants = {
        'Q1 top-left':     [(r, c) for r in range(0, half) for c in range(0, half)],
        'Q2 top-right':    [(r, c) for r in range(0, half) for c in range(half, n)],
        'Q3 bottom-left':  [(r, c) for r in range(half, n) for c in range(0, half)],
        'Q4 bottom-right': [(r, c) for r in range(half, n) for c in range(half, n)],
    }

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (name, cells) in zip(axes, quadrants.items()):
        for (r, c) in cells:
            ax.plot(cell_traces[(r, c)], label=f'cell ({r},{c})')
        ax.set_title(name)
        ax.legend(fontsize=7)

    fig.suptitle(fig_title)
    fig.tight_layout()
    return fig


def binarized_traces(data, binarized_mask):
    """
    binarized_mask: (y, x) with values 255 (bright) / 0 (dark)
    Returns dF/F traces for the bright region and the dark region.
    """
    bright_region = binarized_mask == 255
    dark_region = binarized_mask == 0

    bright_pixels = data[:, bright_region]
    bright_total = bright_pixels.mean(axis=1)
    bright_dff = (bright_total - np.median(bright_total)) / np.median(bright_total)

    dark_pixels = data[:, dark_region]
    dark_total = dark_pixels.mean(axis=1)
    dark_dff = (dark_total - np.median(dark_total)) / np.median(dark_total)

    return bright_dff, dark_dff


if __name__ == "__main__":
    input_folder = "C:\\Users\\megan\\flies\\activity_analysis"
    video_folders = get_video_folders(input_folder)

    for vf in video_folders:
        folder = vf['folder']
        base = vf['base']

        video_path = os.path.join(folder, f"{base}.tif")
        avg_path = os.path.join(folder, f"AVG_{base}.tif")
        clahe_path = os.path.join(folder, f"AVG_{base}_clahe.npy")
        binarized_path = os.path.join(folder, f"AVG_{base}_binarized.npy")
        neuropil_path = os.path.join(folder, f"{base}_neuropils.tif")

        data = tifffile.imread(video_path)
        avg_data = tifffile.imread(avg_path)
        clahe_image = np.load(clahe_path)
        binarized = np.load(binarized_path)
        neuropils = tifffile.imread(neuropil_path)

        title = base

        total = np.mean(data, axis=(1, 2))
        total = (total - np.median(total)) / np.median(total)

        label_to_name = label_neuropils(neuropils)
        neuropil_dff = neuropil_traces(data, neuropils, label_to_name)

        bright, dark = binarized_traces(data, binarized)

        # one combined plot per video: total activity, all neuropils, and
        # binarized bright/dark all overlaid together as separate lines
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(total, label='total', color='black', linewidth=1.8)
        for name, dff in neuropil_dff.items():
            ax.plot(dff, label=name, linewidth=1.2)
        ax.plot(bright, label='bright', linestyle='--', linewidth=1.2)
        ax.plot(dark, label='dark', linestyle='--', linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel('frame')
        ax.set_ylabel('dF/F')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, f"{base}_activity.svg"))
        plt.close(fig)

        # quadrant activity - saved into this video's own subfolder
        cell_traces = grid_traces(data, n=4)
        quad_fig = quadrant_plot(title, cell_traces, n=4)
        quad_fig.savefig(os.path.join(folder, f"{base}_quadrants.svg"))
        plt.close(quad_fig)

        print(f'activity done: {folder}')