import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from himalaya.scoring import correlation_score
from himalaya.ridge import Ridge
import glob
import utils

import time


# read pickle
def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

# write pickle
def write_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def compute_neural_predictivity(brain_feats, model_feats, subj):
    X_train, X_test = [], []
    Y_train, Y_test = [], []
    
    train_ids = read_pickle(f'/SSD/slava/THESIS/NSD_processed/subj{subj}_train_nsd-img_to_fmri.pkl')
    
    for k, v in brain_feats.items():
        if k in train_ids:
            Y_train.append(v)
            X_train.append(model_feats[k].numpy())
        else:
            Y_test.append(v)
            X_test.append(model_feats[k].numpy())
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    
    print(f'X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}')
    print(f'X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}')
    
    alignment_score = {}
    
    for i in range(X_train.shape[1]):
        model = Ridge(alpha=60000)
        
        model.fit(X_train[:, i, :], Y_train)
        corr_score = correlation_score(model.predict(X_test[:, i, :]), Y_test)
        
        alignment_score[(0, i)] = corr_score
    
    return alignment_score
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Measure alignment between brain and DNN models')
    parser.add_argument('--subj', type=int, default=1, help='Subject number')
    parser.add_argument('--roi', type=str, default='general', help='Which ROI of brain to use as brain features')
    parser.add_argument('--modality', type=str, required=True)
    parser.add_argument('--model_family', type=str, required=True)
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument("--metric", type=str, default="neural_predictivity")
    parser.add_argument("--is_channel", action='store_true', help='Whether to use channel-wise extracted features')
    args = parser.parse_args()
    
    roi_arr = read_pickle(f'./convergence_data/subj0{args.subj}/{args.roi}/img_to_features.pkl')
    model_names = utils.get_models(args)
    
    save_path = f'./alignment_scores/subj0{args.subj}/{args.roi}/{args.modality}/{args.model_family}_{args.metric}.pkl'
    os.makedirs(f'./alignment_scores/subj0{args.subj}/{args.roi}/{args.modality}', exist_ok=True)
    
    if os.path.exists(save_path):
        print(f"File {save_path} already exists.")
        alignment_results = read_pickle(save_path)
    else:
        alignment_results = {}
        
    for model in model_names:
        if model in alignment_results:
            print(f"Alignment for {model} already computed.")
            print(f"Alignment scores for {model}: ", alignment_results[model]["score"])
            print(f"Alignment indices for {model}: ", alignment_results[model]["indices"])
            print('-'*100)
            continue
        
        print(f"Computing alignment for {model}")
        start = time.time()
        
        if args.modality == 'vision':
            model_feats = torch.load(f'./vision_features/{model}/stim_features_{args.pooling}.pt', map_location='cpu')
            
            num_params = model_feats['num_params']
            model_feats = model_feats['feats']
                
        elif args.modality == 'language':
            if args.is_channel:
                model_feats = sorted(glob.glob(f'./language_features/{model}/stim_features_{args.pooling}_channel_*.pt'),
                                     key=lambda x: int(x.split('_')[-1].split('.')[0]))
                print("Found_channels ", len(model_feats))
            else:
                model_feats = torch.load(f'./language_features/{model}/stim_features_{args.pooling}.pt', map_location='cpu')
            
                num_params = model_feats['num_params']
                model_feats = model_feats['feats']
        
        if type(model_feats) == dict: # for vision models
            model_feats = {k: v.unsqueeze(1) for k, v in model_feats.items()}
        elif type(model_feats) == list: # for language models per channel
            model_feats = {i: v for i, v in enumerate(model_feats)}
        else:
            model_feats = {'all': model_feats}
        
        for k, v in model_feats.items():
            if not args.is_channel:
                if args.metric == 'neural_predictivity':
                    alignment_scores = compute_neural_predictivity(roi_arr, model_feats[k], args.subj)
                else:
                    raise ValueError(f"Unknown metric: {args.metric}")
            else:
                feats_ = torch.load(v, map_location='cpu')
                num_params = feats_['num_params']
                feats_ = feats_['feats'].unsqueeze(1)
                
                alignment_scores = compute_neural_predictivity(roi_arr, feats_, args.subj)
                print(f"Alignment scores for {model}-{k}-{args.pooling}: ", alignment_scores)
                
                if f'{model}_all_{args.pooling}' in alignment_results:
                    alignment_results[f'{model}_all_{args.pooling}']["score"][(0, k)] = alignment_scores[(0, 0)]
                else:
                    alignment_results[f'{model}_all_{args.pooling}'] = {"score": alignment_scores, 
                                                                        "num_params": num_params}
                
                elapsed = time.time() - start
                minutes, seconds = divmod(int(elapsed), 60)
                print(f"Time taken: {minutes} minutes and {seconds} seconds")
                
                print('-'*100)
            
                # save pickle
                write_pickle(alignment_results, save_path)
            
                continue
            

            print(f"Alignment scores for {model}-{k}-{args.pooling}: ", alignment_scores)
            
            alignment_results[f'{model}_{k}_{args.pooling}'] = {"score": alignment_scores, "num_params": num_params}
            
            elapsed = time.time() - start
            minutes, seconds = divmod(int(elapsed), 60)
            print(f"Time taken: {minutes} minutes and {seconds} seconds")
            
            print('-'*100)
        
            # save pickle
            write_pickle(alignment_results, save_path)