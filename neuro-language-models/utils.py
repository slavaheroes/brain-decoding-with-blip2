import os

import yaml
import nibabel as nib
import pickle

import torch
import numpy as np
from loguru import logger
from tqdm import tqdm

from collections import OrderedDict

def read_yaml(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def read_pickle(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pickle(data, path):
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def get_fmri_arr(subj, fmri_id, split='train'):
    fmri_path = f'/SSD2/guest/slava/THESIS/NSD_processed/subj0{subj}/fMRI_frames_normalized/fmri_{fmri_id}_{split}.npy'
    assert os.path.exists(fmri_path), f"File {fmri_path} does not exist"
    return fmri_path

def build_dataset(ids, fmri_memory, test_ids, subj, config, split='train'):
    
    X = []
    Y = []
    
    mem_size = config['DATA']['memory_size']
    for fmri_id, nsd_img in ids.items():
        memory = fmri_memory[fmri_id]
        past = list(memory['past'])
        future = list(memory['future'])
        
        fmris = []
        
        if config['DATA']['memory_type']=='past_only':
            for past_id in past[:mem_size-1][::-1]:
                fmris.append(past_id)
            fmris.append(fmri_id)
            
        elif config['DATA']['memory_type']=='future_only':
            fmris.append(fmri_id)
            for future_id in future[:mem_size-1]:
                fmris.append(future_id)
            
        elif config['DATA']['memory_type']=='both':
            for past_id in past[:(mem_size-1)//2][::-1]:
                fmris.append(past_id)
                
            fmris.append(fmri_id)
            if (mem_size-1)%2==1:
                fut = future[:(mem_size-1)//2 + 1]
            else:
                fut = future[:(mem_size-1)//2]
                
            for future_id in fut:
                fmris.append(future_id)
                
        else:
            raise ValueError(f"Unknown memory type {config['DATA']['memory_type']}")
        
        assert len(fmris) == mem_size, f"Expected {mem_size} fmris, got {len(fmris)}"
        
        # extract paths
        img = os.path.join(config['DATA']['path_to_vision_features'], f'nsd-{nsd_img}.pt')
        assert os.path.exists(img), f"File {img} does not exist"
        
        fmri_paths = []
        
        for i in fmris:
            if i < 0 or i >= (750*37):
                fmri_paths.append(None)
            elif split=='train' and i in test_ids:
                fmri_paths.append(None)
            else:
                if i in test_ids:
                    fmri_paths.append(get_fmri_arr(subj, i, 'test'))
                else:
                    fmri_paths.append(get_fmri_arr(subj, i, 'train'))
        
        X.append(fmri_paths)
        Y.append(img)
    
    return X, Y

def load_fmri(paths, shape, roi=None):
    fmris = []
    for path in paths:
        if path is None:
            arr = np.zeros(shape)
        else:
            arr = np.load(path)
            assert arr.shape[0] == 1, f"Expected shape {shape}, got {arr.shape}"
            arr = arr[0]
        
        if roi is not None:
            arr = arr[roi>0]
        fmris.append(arr)
    
    return np.concatenate(fmris)

def load_data(subj, config):
    
    if config['DATA']['memory_size'] > 1:
        # fmri-id to nsd-img mapping
        train_idx = read_pickle(f'/SSD2/guest/slava/THESIS/NSD_processed/subj{subj}_train_mri_to_nsd-img.pkl')
        test_idx = read_pickle(f'/SSD2/guest/slava/THESIS/NSD_processed/subj{subj}_test_mri_to_nsd-img.pkl')
        
        fmri_memory = read_pickle(f'/SSD2/guest/slava/THESIS/NSD_processed/subj{subj}_fmri_sequence_memory.pkl')
        
        train_X, train_Y = build_dataset(train_idx, fmri_memory, 
                                        test_idx, subj, config, split='train')
        test_X, test_Y = build_dataset(test_idx, fmri_memory, 
                                    test_idx, subj, config, split='test')
    else:
        # nsd-img to fmri-id mapping
        train_idx = read_pickle(f'/SSD2/guest/slava/THESIS/NSD_processed/subj{subj}_train_nsd-img_to_fmri.pkl')
        test_idx = read_pickle(f'/SSD2/guest/slava/THESIS/NSD_processed/subj{subj}_test_nsd-img_to_fmri.pkl')
        
        train_X, train_Y = [], []
        for nsd_img, fmri_id_list in train_idx.items():
            train_X.append([get_fmri_arr(subj, fmri_id, 'train') for fmri_id in fmri_id_list])
            train_Y.append(os.path.join(config['DATA']['path_to_vision_features'], f'nsd-{nsd_img}.pt'))
        
        test_X, test_Y = [], []
        for nsd_img, fmri_id_list in test_idx.items():
            test_X.append( [get_fmri_arr(subj, fmri_id, 'test') for fmri_id in fmri_id_list] )
            test_Y.append(os.path.join(config['DATA']['path_to_vision_features'], f'nsd-{nsd_img}.pt'))
    
    
    logger.info(f"Loaded data for subject {subj}")
    logger.info(f'Length of train_X: {len(train_X)}, train_Y: {len(train_Y)}')
    logger.info(f'Length of test_X: {len(test_X)}, test_Y: {len(test_Y)}')
    
    logger.info(f"Example train_X: {train_X[0]}")
    logger.info(f"Example train_Y: {train_Y[0]}")
    
    return train_X, train_Y, test_X, test_Y
    
    
def load_data_as_array(train_X, train_Y, test_X, test_Y, subj, config, channel=None):
    assert config['DATA']['use_roi'], "ROI is not used but data type is array"
    general_roi = nib.load(f'/SSD2/guest/slava/THESIS/NSD_processed/subj0{subj}/roi/nsdgeneral.nii.gz').get_fdata()
    logger.info(f"Loaded general ROI for subject {subj} with shape {general_roi.shape}")
    
    if config['DATA']['memory_size'] > 1:
        arr_X_train, arr_Y_train = [], []
        arr_X_test, arr_Y_test = [], []
        
        for X, Y in tqdm(zip(train_X, train_Y), total=len(train_X)):
            arr_X_train.append(load_fmri(X, general_roi.shape, general_roi))
            if channel is not None:
                arr_Y_train.append(torch.load(Y, map_location='cpu').numpy()[channel, :])
            else:
                arr_Y_train.append(torch.load(Y, map_location='cpu').numpy())
        
        arr_X_train = np.array(arr_X_train)
        arr_Y_train = np.array(arr_Y_train)
        logger.info(f"Loaded train data with shapes {arr_X_train.shape}, {arr_Y_train.shape}")
        
        for X, Y in tqdm(zip(test_X, test_Y), total=len(test_X)):
            arr_X_test.append(load_fmri(X, general_roi.shape, general_roi))
            if channel is not None:
                arr_Y_test.append(torch.load(Y, map_location='cpu').numpy()[channel, :])
            else:
                arr_Y_test.append(torch.load(Y, map_location='cpu').numpy())
        
        arr_X_test = np.array(arr_X_test)
        arr_Y_test = np.array(arr_Y_test)
        logger.info(f"Loaded test data with shapes {arr_X_test.shape}, {arr_Y_test.shape}")
    else:
        arr_X_train, arr_Y_train = [], []
        arr_X_test, arr_Y_test = [], []
        
        for X, Y in tqdm(zip(train_X, train_Y), total=len(train_X)):
                        
            arr_X_train.append(
                np.array([ load_fmri([x], general_roi.shape, general_roi) for x in X ]).mean(0)
            )
            
            if channel is not None:
                arr_Y_train.append(torch.load(Y, map_location='cpu').numpy()[channel, :])
            else:
                arr_Y_train.append(torch.load(Y, map_location='cpu').numpy())
        
        arr_X_train = np.array(arr_X_train)
        arr_Y_train = np.array(arr_Y_train) 
        logger.info(f"Loaded train data with shapes {arr_X_train.shape}, {arr_Y_train.shape}")
        
        for X, Y in tqdm(zip(test_X, test_Y), total=len(test_X)):
            arr_X_test.append(
                np.array([load_fmri([x], general_roi.shape, general_roi) for x in X]).mean(0)
            )
            
            if channel is not None:
                arr_Y_test.append(torch.load(Y, map_location='cpu').numpy()[channel, :])
            else:
                arr_Y_test.append(torch.load(Y, map_location='cpu').numpy())
        
        arr_X_test = np.array(arr_X_test)
        arr_Y_test = np.array(arr_Y_test)
        
        logger.info(f"Loaded test data with shapes {arr_X_test.shape}, {arr_Y_test.shape}")
    
    return arr_X_train, arr_Y_train, arr_X_test, arr_Y_test

def format_pl_state_dict(state_dict, prefix='model.'):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def batchwise_cosine_similarity(Z,B):
    Z = Z.flatten(1)
    B = B.flatten(1).T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def topk(similarities,labels,k=5):    
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum