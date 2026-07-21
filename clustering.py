import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.ndimage import gaussian_filter, zoom
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
from scipy.signal import find_peaks
from sklearn.metrics import silhouette_samples


def pixel_findroi(data, filename, output="clustered"):
    start = time.perf_counter()
    os.makedirs(output, exist_ok=True)

    t, y_size, x_size = data.shape
    num_pixels = y_size * x_size

    pixel_traces = data.reshape(t, num_pixels).T  # (num_pixels, t)

    # compute stats needed to identify unusable pixels
    stddev = pixel_traces.std(axis=1)
    f0_all = np.median(pixel_traces, axis=1)

    # exclude pixels that are flat (no variance) OR have zero median (would divide by zero)
    valid_mask = (stddev > 0) & (f0_all != 0)
    n_dead = int((~valid_mask).sum())
    if n_dead > 0:
        print(f"  excluding {n_dead} flat/zero-median pixels from clustering")

    valid_indices = np.where(valid_mask)[0]
    num_features = len(valid_indices)

    valid_traces = pixel_traces[valid_mask]
    f0 = f0_all[valid_mask]

    deltaf = (valid_traces - f0[:, None]) / f0[:, None]

    # remove the shared/global signal component (e.g. a whole-frame stimulus
    # response present in nearly every pixel) before clustering. Without this,
    # correlation-based clustering and silhouette scoring are dominated by
    # whatever large shared event all pixels have in common, making genuinely
    # different pixels look similar and making "noise" pixels resemble whichever
    # real cluster the shared component happens to look like.
    global_trace = np.mean(deltaf, axis=0)                 # (t,) - average across all pixels
    residual = deltaf - global_trace[None, :]              # local, pixel-specific signal only

    # corrcoef and distance matrices - computed on the residual, not raw deltaf,
    # so clustering is driven by local differences rather than the shared component
    corrcoef_matrix = np.corrcoef(residual)
    distance = 1 - corrcoef_matrix
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)

    # hierarchical clustering
    linkage_matrix = linkage(squareform(distance), method='average')
    cluster_order = leaves_list(linkage_matrix)
    corrcoef_reordered = np.corrcoef(residual[cluster_order])

    # find best k using mean silhouette score
    min_clusters = 2
    all_scores = {}

    for k in range(min_clusters, int(min(50, np.ceil(num_features / 2)))):
        labels = fcluster(linkage_matrix, t=k, criterion='maxclust') - 1
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_samples(distance, labels, metric='precomputed').mean()
        all_scores[k] = score

    sorted_ks = sorted(all_scores.keys())
    scores_arr = np.array([all_scores[k] for k in sorted_ks])

    # find genuine local maxima (peaks that rise meaningfully above surrounding valleys),
    # rather than small noisy wiggles in an overall decreasing trend
    prominence_threshold = 0.01  # tune this: larger = only very pronounced bumps count
    peak_idxs, properties = find_peaks(scores_arr, prominence=prominence_threshold)

    if len(peak_idxs) > 0:
        # among genuine peaks, pick the one with the highest silhouette score
        best_peak_idx = peak_idxs[np.argmax(scores_arr[peak_idxs])]
        best_k = sorted_ks[best_peak_idx]
    else:
        # no clear peak found (e.g. purely monotonic curve) -> fall back to global max
        best_k = max(all_scores, key=all_scores.get)

    best_raw_labels = fcluster(linkage_matrix, t=best_k, criterion='maxclust') - 1
    best_score = all_scores[best_k]

    noise_threshold = 0.0
    min_cluster_size = 10

    per_roi_scores = silhouette_samples(distance, best_raw_labels, metric='precomputed')
    is_noise = per_roi_scores < noise_threshold

    # also mark any pixel belonging to a cluster smaller than min_cluster_size as noise
    raw_cluster_sizes = np.bincount(best_raw_labels)
    small_cluster_mask = raw_cluster_sizes[best_raw_labels] < min_cluster_size
    n_small_cluster_pixels = int((small_cluster_mask & ~is_noise).sum())
    is_noise = is_noise | small_cluster_mask

    n_real = len(np.unique(best_raw_labels))
    cluster_labels = best_raw_labels.copy()
    next_noise_label = n_real
    for i in range(num_features):
        if is_noise[i]:
            cluster_labels[i] = next_noise_label
            next_noise_label += 1
    n_clusters = next_noise_label
    noise_count = int(is_noise.sum())

    real_cluster_sizes = [(i, len(np.where(cluster_labels == i)[0])) for i in range(n_real)]
    real_cluster_sizes.sort(key=lambda x: x[1], reverse=True)

    label_remap = {}
    for new_idx, (old_idx, _) in enumerate(real_cluster_sizes):
        label_remap[old_idx] = new_idx
    cluster_labels = np.array([label_remap.get(l, l) for l in cluster_labels])

    set1 = [plt.cm.Set1(i) for i in range(8)]
    set3 = [plt.cm.Set3(i) for i in range(12)]
    all_colors = set1 + set3
    colors = []
    for i in range(n_clusters):
        if i < n_real:
            colors.append(all_colors[i % len(all_colors)])
        else:
            colors.append((0.3, 0.3, 0.3, 1.0))

    print(f"\nClustering results for {filename}:")
    print(f"  best k={best_k} (mean silhouette={best_score:.4f})")
    print(f"  clusters: {n_real}  |  noise pixels: {noise_count} (incl. {n_small_cluster_pixels} from clusters <{min_cluster_size}px)  |  total clustered: {num_features}  |  excluded dead: {n_dead}")

    groups = [[] for _ in range(n_clusters)]
    for cluster_idx in range(n_clusters):
        pixel_indices = np.where(cluster_labels == cluster_idx)[0]
        for pixel_idx in pixel_indices:
            groups[cluster_idx].append(pixel_idx)

    active_real_clusters = [i for i in range(n_real) if np.sum(cluster_labels == i) > 0]
    n_active = len(active_real_clusters)
    n_plots = n_active + (1 if noise_count > 0 else 0)

    plt.figure(1, figsize=(20, max(15, n_plots * 0.5)))
    for plot_pos, real_idx in enumerate(active_real_clusters):
        pixel_indices = np.where(cluster_labels == real_idx)[0]
        plt.subplot(n_plots, 1, plot_pos + 1)
        avg_trace = np.mean(deltaf[pixel_indices], axis=0)
        plt.plot(avg_trace, color=colors[real_idx], linewidth=1.5)
        plt.ylabel(f'C{plot_pos+1}\n(n={len(pixel_indices)})', fontsize=7, rotation=0, labelpad=35)
        is_last = (plot_pos == n_plots - 1)
        plt.tick_params(labelbottom=is_last)

    if noise_count > 0:
        noise_indices = np.where(cluster_labels >= n_real)[0]
        plt.subplot(n_plots, 1, n_plots)
        avg_trace = np.mean(deltaf[noise_indices], axis=0)
        plt.plot(avg_trace, color=(0.3, 0.3, 0.3, 1.0), linewidth=1.5)
        plt.ylabel(f'Noise\n(n={len(noise_indices)})', fontsize=7, rotation=0, labelpad=35)
        plt.tick_params(labelbottom=True)

    plt.figure(1).savefig(os.path.join(output, f"{filename}_clusters_over_time.svg"))
    plt.close(plt.figure(1))

    # color coded spatial map: map back through valid_indices to original (row, col)
    group_colors = np.zeros((y_size, x_size, 3))
    for group_idx, group in enumerate(groups):
        color = colors[group_idx][:3]
        for local_idx in group:
            original_idx = valid_indices[local_idx]  # map back to full 4096-pixel index
            row = original_idx // x_size
            col = original_idx % x_size
            group_colors[row, col] = color
    # dead pixels stay black (0,0,0) since they're never assigned a color

    plt.figure(2, figsize=(20, 15))
    plt.subplot(1, 2, 1)
    plt.imshow(group_colors)
    plt.title("Clusters (gray=noise, black=excluded)")

    plt.subplot(1, 2, 2)
    plt.imshow(corrcoef_reordered, cmap='Spectral', vmin=-1, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.figure(2).savefig(os.path.join(output, f"{filename}_groups_and_matrix.svg"))
    plt.close(plt.figure(2))

    # silhouette plots
    fig_sil, axes_sil = plt.subplots(1, 2, figsize=(14, 4))
    fig_sil.suptitle(f"Silhouette analysis — {filename}", fontsize=11)
    ks = sorted(all_scores.keys())
    axes_sil[0].plot(ks, [all_scores[k] for k in ks], marker='o', color="steelblue", linewidth=1.5)
    axes_sil[0].axvline(best_k, color="tomato", linewidth=1.2, linestyle="--",
                        label=f"best k={best_k} ({best_score:.4f})")
    axes_sil[0].set_xlabel("k")
    axes_sil[0].set_ylabel("Mean silhouette score")
    axes_sil[0].set_title("Silhouette score vs k")
    axes_sil[0].set_xticks(ks)
    axes_sil[0].legend(fontsize=8)
    roi_idx_arr = np.arange(num_features)
    bar_colors = [colors[cluster_labels[i]][:3] for i in roi_idx_arr]
    axes_sil[1].bar(roi_idx_arr, per_roi_scores, color=bar_colors, edgecolor="none")
    axes_sil[1].axhline(noise_threshold, color="tomato", linewidth=1.2, linestyle="--",
                        label=f"noise threshold ({noise_threshold})")
    axes_sil[1].set_xlabel("Pixel index")
    axes_sil[1].set_ylabel("Silhouette score")
    axes_sil[1].set_title("Per-pixel silhouette (gray=noise)")
    axes_sil[1].set_xticks([])
    axes_sil[1].legend(fontsize=8)
    plt.tight_layout()
    fig_sil.savefig(os.path.join(output, f"{filename}_silhouette.svg"))
    plt.close(fig_sil)

    end = time.perf_counter()
    print(f'time: {(end - start):.2f} seconds')


if __name__ == "__main__":
    input_folder = "C:\\Users\\megan\\flies\\activity_analysis"
    tif_videos = [f'{input_folder}\\registered\\{f}' for f in sorted(os.listdir(f'{input_folder}\\registered')) if f.endswith(".tif")]
    os.makedirs('clustered', exist_ok=True)

    for i in range(4):
        data = zoom(gaussian_filter(tifffile.imread(tif_videos[i]), sigma=(0, 1, 1)),
                    zoom=(1, 0.25, 0.25), order=1, prefilter=True)

        filename = os.path.splitext(os.path.basename(tif_videos[i]))[0]
        pixel_findroi(data, filename=filename, output='clustered')
        print(f'pixel clustering done: {filename}')