# remove pixels lost to motion registration, and organize each newly
# registered video into its own folder. This is the ONLY script that ever
# creates a video's folder -- every other script just discovers folders
# that already exist.
import os
import numpy as np
import tifffile

input_folder = "C:\\Users\\megan\\flies\\activity_analysis"

# folder where the motion-registration tool drops its finished output
# (adjust this if your registered files land somewhere else)
done_folder = os.path.join(input_folder, "raw data", "done")


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
        candidates = [
            f for f in os.listdir(entry.path)
            if f.endswith('_registered.tif') and not f.startswith('AVG_')
        ]
        if not candidates:
            continue
        base = os.path.splitext(candidates[0])[0]
        folders.append({'folder': entry.path, 'base': base})
    return folders


if __name__ == "__main__":
    registered_files = [f for f in sorted(os.listdir(done_folder)) if f.endswith(".tif")]

    for f in registered_files:
        base = os.path.splitext(f)[0]  # e.g. 'fly1_less_than24hr_pre_dark_XY0_Z0_T0000_C0'
        source_path = os.path.join(done_folder, f)

        # create this video's own folder - the only place folder creation happens
        video_folder = os.path.join(input_folder, base)
        os.makedirs(video_folder, exist_ok=True)

        data = tifffile.imread(source_path)  # (t, y, x)

        # get rid of pixels that were 0 for any point in time
        background_mask = np.any(data == 0, axis=0)  # shape: (y, x)
        data[:, background_mask] = 0

        # write the processed video into its new folder, with the
        # '_registered' suffix expected by the rest of the pipeline
        dest_path = os.path.join(video_folder, f"{base}_registered.tif")
        tifffile.imwrite(dest_path, data)

        # remove the original from raw data/done now that the processed
        # version has been moved into its own folder
        os.remove(source_path)

        print(f'moved + background removed: {source_path} -> {dest_path}')