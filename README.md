# activity_analysis
analysis of 4 files to get activity

update: changed folder structure
    - each file has their own folder for inputs and outputs
    - removebg takes the registered from raw data and moves it to this new folder
    - total + neuropils + binarized activity is now in the same plot per video

pipeline: register -> removebg.py -> fiji avg -> clahe_binarize.py -> napari neuropils -> activity.py
                                  -> clustering.py (only needs video)

---

update: clustering.py
    - same as findrois.py in roisfromactivity, but for pixels
    - it now subtracts the global trace to compute the differences better
    - min clusters is 2, max clusters is 50
    - min 30 pixels to count as a cluster
    - silhouette score peak is by scipy.signal.find_peaks with prominence 0.001

---

step 1: just get the fluorescence of the entire thing
    - done, but what else do i do with this information
    - also removed pixels lost to motion registration
    - this has been done in a separate file, and i redid avg and clahe
step 2: segment into neuropils
    - stuck here, how would i segment
    - probably just draw it on napari and import as npy mask, but hard to see
step 3: segment a bit more
    - this actually feels like it might be easier bc i can see layers

pipeline: register -> removebg.py -> fiji avg -> clahe.py