import argparse
import os

import numpy as np

import pickle
import nibabel as nib

# read pickle
def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

# write pickle
def write_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def extract_roi_mask(subj, roi_name):
    """
    Extracts the mask of a given ROI for a given subject.
    
    Parameters:
    -----------
    subj: int
        Subject number.
    roi: str
        ROI name.
    Returns:
    --------
    roi_mask: np.ndarray
        ROI mask.
    """
    nsdgeneral_mask = nib.load(f'/SSD/slava/THESIS/NSD_processed/subj0{subj}/roi/nsdgeneral.nii.gz').get_fdata()


    if 'visual' in roi_name:
        visual_roi = nib.load(f'/SSD/slava/THESIS/NSD_processed/subj0{subj}/roi/prf-visualrois.nii.gz').get_fdata()
        
        if 'V1' in roi_name:
            roi = np.zeros_like(nsdgeneral_mask)
            roi[visual_roi==1] = 1
            roi[visual_roi==2] = 1
        elif 'V2' in roi_name:
            roi = np.zeros_like(nsdgeneral_mask)
            roi[visual_roi==3] = 1
            roi[visual_roi==4] = 1
        elif 'V3' in roi_name:
            roi = np.zeros_like(nsdgeneral_mask)
            roi[visual_roi==5] = 1
            roi[visual_roi==6] = 1
        elif 'V4' in roi_name:
            roi = np.zeros_like(nsdgeneral_mask)
            roi[visual_roi==7] = 1
        else:
            # all visual
            roi = visual_roi
    elif roi_name=='floc-words':
        roi = nib.load(f'/SSD/slava/THESIS/NSD_processed/subj0{subj}/roi/floc-words.nii.gz').get_fdata()
    elif roi_name=='floc-faces':
        roi = nib.load(f'/SSD/slava/THESIS/NSD_processed/subj0{subj}/roi/floc-faces.nii.gz').get_fdata()
    elif roi_name=='floc-places':
        roi = nib.load(f'/SSD/slava/THESIS/NSD_processed/subj0{subj}/roi/floc-places.nii.gz').get_fdata()
    elif roi_name=='floc-bodies':
        roi = nib.load(f'/SSD/slava/THESIS/NSD_processed/subj0{subj}/roi/floc-bodies.nii.gz').get_fdata()
    elif roi_name=='general':
        roi = nsdgeneral_mask
    else:
        raise ValueError('Unknown roi_name')

    # roi = roi[nsdgeneral_mask>0]
    roi[roi<0] = 0
    roi[roi>0] = 1
    
    return roi

def extract_fmri_arr(fmri_ids: list, subj: int, split: str):
    fmri_arr = []
    
    for f_id in fmri_ids:
        f = np.load(f'/SSD/slava/THESIS/NSD_processed/subj0{subj}/fMRI_frames_normalized/fmri_{f_id}_{split}.npy')
        fmri_arr.append(f)
    
    return np.array(fmri_arr).mean(axis=0)


if __name__=='__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--subj', type=int, required=True)
    args = parser.parse_args()
    
    subj = args.subj
    
    rois = ['visual-V1', 'visual-V2', 'visual-V3', 'visual-V4', 'visual',
            'floc-faces', 'floc-places', 'floc-bodies', 'floc-words', 'general']
    
    train_img_to_fmri = read_pickle(f'/SSD/slava/THESIS/NSD_processed/subj{subj}_train_nsd-img_to_fmri.pkl')
    test_img_to_fmri = read_pickle(f'/SSD/slava/THESIS/NSD_processed/subj{subj}_test_nsd-img_to_fmri.pkl')
    
    print("Length of train_img_to_fmri: ", len(train_img_to_fmri))
    print("Length of test_img_to_fmri: ", len(test_img_to_fmri))
    
    for roi in rois:
        print(f"Extracting features for {roi}")
        roi_mask = extract_roi_mask(subj, roi)
        
        # for each ROI, save the extracted signals
        os.makedirs(f'./convergence_data/subj0{subj}/{roi}', exist_ok=True)
        
        img_to_features = {}
        
        for nsd_img_id, fmri_ids in train_img_to_fmri.items():
            fmri_arr = extract_fmri_arr(fmri_ids, subj, 'train')
            assert fmri_arr.shape[0]==1 and len(fmri_arr.shape)==4, "Incorrect shape"
            fmri_arr = fmri_arr[0]
            
            feat = fmri_arr[roi_mask>0]
            
            assert nsd_img_id not in img_to_features, "nsd_img_id already exists"
            img_to_features[nsd_img_id] = feat
                        
        for nsd_img_id, fmri_ids in test_img_to_fmri.items():
            fmri_arr = extract_fmri_arr(fmri_ids, subj, 'test')
            assert fmri_arr.shape[0]==1 and len(fmri_arr.shape)==4, "Incorrect shape"
            fmri_arr = fmri_arr[0]
            
            feat = fmri_arr[roi_mask>0]
            
            assert nsd_img_id not in img_to_features, "nsd_img_id already exists"
            img_to_features[nsd_img_id] = feat
                    
        write_pickle(img_to_features, f'./convergence_data/subj0{subj}/{roi}/img_to_features.pkl')
        print(f"Saved features for {roi}")
              