# remove pixels lost to motion registration, and organize each newly registered video into its own folder. 
import os
import numpy as np
import tifffile

input_folder = "C:\\Users\\megan\\flies\\activity_analysis"

# folder where the motion-registration tool drops its finished output
done_folder = os.path.join(input_folder, "raw data")


def get_video_folders(input_folder):
    """
    Each video lives in its own subfolder (named after the raw file),
    containing '{base}_registered.tif' plus all of that video's related
    outputs (AVG image, clahe/binarized masks, neuropil mask, plots).
    Returns a list of dicts: {'folder': subfolder path, 'base': filename
    base, e.g. 'fly1_..._registered', without the .tif extension}.
    """
    folders = []
    for entry in sorted(os.scandir(input_folder), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        expected_file = f"{entry.name}_registered.tif"
        if not os.path.isfile(os.path.join(entry.path, expected_file)):
            continue
        base = os.path.splitext(expected_file)[0]
        folders.append({'folder': entry.path, 'base': base})
    return folders


if __name__ == "__main__":
    registered_files = [f for f in sorted(os.listdir(done_folder)) if f.endswith("_registered.tif")]

    for f in registered_files:
        base = f[:-len("_registered.tif")]
        source_path = os.path.join(done_folder, f)

        # create this video's own folder, named after the plain filename
        video_folder = os.path.join(input_folder, base)
        os.makedirs(video_folder, exist_ok=True)

        data = tifffile.imread(source_path)  # (t, y, x)

        # get rid of pixels that were 0 for any point in time
        background_mask = np.any(data == 0, axis=0)  # (y, x)
        data[:, background_mask] = 0

        # write the processed video into its new folder
        dest_path = os.path.join(video_folder, f"{base}_registered.tif")
        tifffile.imwrite(dest_path, data)

        # remove original registered from raw data
        os.remove(source_path)

        print(f'moved + background removed: {source_path} -> {dest_path}')