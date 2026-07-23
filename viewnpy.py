import glob
import os
import numpy as np

# Set your target folder path
folder_path = "C:\\Users\\megan\\flies\\activity_analysis\\raw data"

# Find all matching files
file_paths = glob.glob(os.path.join(folder_path, "*ops.npy"))

# Loop through and view data
for path in file_paths:
    print(f"\n--- File: {os.path.basename(path)} ---")
    data = np.load(path, allow_pickle=True)
    print(data)
