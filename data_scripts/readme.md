# Data Preperation

### Downloading

Run `download_nsddata.py` script that was borrowed from [brain-diffuser](https://github.com/ozcelikfu/brain-diffuser) repository.

Additionally, you need to download *coco_annotations* and extract files from the archive:

```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

### Organizing the data

Please, run the following Jupyter notebooks in order. While running, change the directories accordingly. 

1. `initial_data_organization.ipynb`

- NSD data is splitted into train/test sets.
- fMRI sequence order is extracted such that for a given fMRI frame we can retrieve the past and future frames.
- NSD-COCO relationship is explored.

2. `fMRI_data_organization.ipynb`

- *Mean/Standard deviation* Statistics are calculated for each subject based on their train set. 
- Each frame of fMRI is normalized and saved as *.npy* file.
- *Nifti* files of ROI masks are copied.

### How the data should look like? 

After all the steps above, the data should be organized as follows:

NSD_processed/
    ├── subj01/
    │   ├── fMRI_frames_normalized/
    │   │   ├── fmri_1_train.npy
    │   │   ├── fmri_0_test.npy
    │   │   └── **...**
    │   ├── roi/
    │   │   ├── nsdgeneral.nii.gz
    │   │   └── **...**
    │   ├── train_mean.npy
    │   └── train_std.npy
    ├── `[similar folders for other subjects]`
    ├── subj1_fmri_sequence_memory.pkl
    ├── subj1_train_mri_to_nsd-img.pkl
    ├── subj1_test_mri_to_nsd-img.pkl
    ├── subj1_train_nsd-img_to_fmri.pkl
    ├── subj1_test_nsd-img_to_fmri.pkl
    ├── **...**
    └── `[similar files for other subjects]`