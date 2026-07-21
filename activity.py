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
    tif_videos = [f'{input_folder}\\registered\\{f}' for f in sorted(os.listdir(f'{input_folder}\\registered')) if f.endswith(".tif")]
    avg_images = [f'{input_folder}\\avg\\{f}' for f in sorted(os.listdir(f'{input_folder}\\avg')) if f.endswith(".tif")]
    clahe_images = [f'{input_folder}\\clahe_binarize\\{f}' for f in sorted(os.listdir(f'{input_folder}\\clahe_binarize')) if f.endswith("clahe.npy")]
    binarized_masks = [f'{input_folder}\\clahe_binarize\\{f}' for f in sorted(os.listdir(f'{input_folder}\\clahe_binarize')) if f.endswith("binarized.npy")]
    neuropil_masks = [f'{input_folder}\\neuropils\\{f}' for f in sorted(os.listdir(f'{input_folder}\\neuropils')) if f.endswith(".tif")]

    os.makedirs('outputs', exist_ok=True)

    fig1, ax1 = plt.subplots(4, 1, figsize=(16,8), sharex=True, sharey=True)
    fig2, ax2 = plt.subplots(4, 1, figsize=(16,8), sharex=True, sharey=True)
    fig3, ax3 = plt.subplots(4, 1, figsize=(16,8), sharex=True, sharey=True)

    for i in range(4):
        data = tifffile.imread(tif_videos[i])
        avg_data = tifffile.imread(avg_images[i])
        clahe_image = np.load(clahe_images[i])
        binarized = np.load(binarized_masks[i])
        neuropils = tifffile.imread(neuropil_masks[i])

        title = os.path.splitext(os.path.basename(tif_videos[i]))[0]

        total = np.mean(data, axis=(1,2))
        total = (total - np.median(total)) / np.median(total)
        ax1[i].plot(total)
        ax1[i].set_title(title)

        label_to_name = label_neuropils(neuropils)
        traces = neuropil_traces(data, neuropils, label_to_name)
        for name, dff in traces.items():
            ax2[i].plot(dff, label=name)
        ax2[i].set_title(title)
        ax2[i].legend()

        # quadrant activity, one figure per video
        cell_traces = grid_traces(data, n=4)
        quad_fig = quadrant_plot(title, cell_traces, n=4)
        quad_fig.savefig(f'outputs/{title}_quadrants.svg')
        plt.close(quad_fig)

        # binarized threshold: bright vs dark activity
        bright, dark = binarized_traces(data, binarized)
        ax3[i].plot(bright, label='bright')
        ax3[i].plot(dark, label='dark')
        ax3[i].set_title(title)
        ax3[i].legend()

    fig1.tight_layout()
    fig1.savefig('outputs/total_activity.svg')

    fig2.tight_layout()
    fig2.savefig('outputs/neuropil_activity.svg')

    fig3.tight_layout()
    fig3.savefig('outputs/binarized_activity.svg')
