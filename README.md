# activity_analysis
analysis of 4 files to get activity

step 1: just get the fluorescence of the entire thing
    - done, but what else do i do with this information
    - also removed pixels lost to motion registration
    - this has been done in a separate file, and i redid avg and clahe
step 2: segment into neuropils
    - stuck here, how would i segment
    - probably just draw it on napari and import as npy mask, but hard to see
step 3: segment a bit more
    - this actually feels like it might be easier bc i can see layers

current pipeline: register -> removebg.py -> fiji avg -> clahe.py