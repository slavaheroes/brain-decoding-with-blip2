import argparse
import os
os.environ['HF_HOME'] = '/SSD/slava/huggingface/'

import h5py
import pickle
from PIL import Image

import math
import numpy as np
import pandas as pd
import ast

import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor

from lm_model import load_llm, load_tokenizer
import utils 

from tqdm import trange

def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
# write pickle
def write_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
def cross_entropy_loss(llm_inputs, llm_outputs):
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    mask = llm_inputs["attention_mask"][:, :-1]
    loss = mask * criterion(
        llm_outputs["logits"][:, :-1].permute(0, 2, 1),
        llm_inputs["input_ids"][:, 1:],
    )
    avg_loss = (loss.sum(-1) / mask.sum(-1))
    return loss, avg_loss


def cross_entropy_to_bits_per_unit(losses, input_strings, unit="byte"):
    """
    Convert cross-entropy losses from nats to bits per byte for each input string.

    Parameters:
    - losses (torch.Tensor): [batch x seq_len] (padding tokens should be 0)
    - input_strings (list of str): List of original input strings.

    Returns:
    - torch.Tensor: Tensor of bits per byte values, one per input string.
    """
    # nats to bits by multiplying with log base 2 of e (since log_e(2) = 1 / log_2(e))
    # sum over the sequence length (total bits for each input string)
    losses_in_bits = (losses.cpu() * torch.log2(torch.tensor(math.e))).sum(1)

    # calculate bytes for each input string and normalize losses (8 bits per character, so roughly num character * 8)
    if unit == "byte":
        bytes_per_input = torch.tensor([len(s.encode('utf-8')) for s in input_strings], dtype=torch.float32)
    elif unit == "char":
        bytes_per_input = torch.tensor([len(s) for s in input_strings], dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported unit {unit}")

    # mormalize by the total number of bytes per input string
    bits_per_byte = losses_in_bits / bytes_per_input
    return bits_per_byte


def extract_vision_features(lvm_model_name, args, stim_images):
    os.makedirs(f'./vision_features/{lvm_model_name}', exist_ok=True)
    save_path = f'./vision_features/{lvm_model_name}/stim_features_{args.pooling}.pt'
    
    if os.path.exists(save_path):
        print(f"Features for {lvm_model_name} with {args.pooling} pooling already exist")
        return
    
    device = f'cuda:{args.gpu}'
    # load model
    vision_model  = timm.create_model(lvm_model_name, pretrained=True).to(device).eval()
    lvm_param_count = sum([p.numel() for p in vision_model.parameters()])
    
    transform = create_transform(
            **resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
        )
    
    if "vit" in lvm_model_name:
        return_nodes = [f"blocks.{i}" for i in range(len(vision_model.blocks))]
    elif 'resnet' in lvm_model_name:
        return_nodes = [f"layer{i}" for i in range(1, 5)]
    elif 'efficientnet' in lvm_model_name:
        return_nodes = [f"blocks.{i}" for i in range(len(vision_model.blocks))]
    elif 'convnext' in lvm_model_name:
        return_nodes = [f"stages.{i}" for i in range(len(vision_model.stages))]
    else:
        raise NotImplementedError(f"unknown model {lvm_model_name}")
    
    vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)
    lvm_feats = {k: [] for k in return_nodes}
    
    for i in trange(0, len(stim_images), args.batch_size):
            with torch.no_grad():                
                ims = torch.stack([transform(Image.fromarray(im)) for im in stim_images[i:i+args.batch_size]]).to(device)
                lvm_output = vision_model(ims)
                
                for k, v in lvm_output.items():
                
                    if args.pooling=='mean':
                        if 'vit' in lvm_model_name:
                            # take the mean of all patch tokens
                            v = v[:, 1:, :].mean(dim=1)
                        else:
                            v = v.mean(dim=[2, 3])
                            assert len(v.shape)==2, "Incorrect shape"
                        lvm_feats[k].append(v.cpu())
                    elif args.pooling=='cls':
                        # take the cls token
                        assert 'vit' in lvm_model_name, "CLS token is only for Vision Transformers"
                        v = v[:, 0, :]
                        lvm_feats[k].append(v.cpu())
                    elif args.pooling=='flatten':
                        v = v.flatten(start_dim=1).cpu()
                        lvm_feats[k].append(v) 
                    else:
                        raise ValueError(f"Unknown pooling method {args.pooling}")
    
    for k in return_nodes:
        lvm_feats[k] = torch.cat(lvm_feats[k])
        print(f"Shape of {k} features: ", lvm_feats[k].shape)
    
    torch.save({"feats": lvm_feats, "num_params": lvm_param_count}, save_path)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    print(f"Features for {lvm_model_name} with {args.pooling} pooling saved at {save_path}")
    
def extract_llm_features(llm_model_name, args, texts):
    os.makedirs(f'./language_features/{llm_model_name}', exist_ok=True)
    save_path = f'./language_features/{llm_model_name}/stim_features_{args.pooling}.pt'
    
    if os.path.exists(save_path):
        print(f"Features for {llm_model_name} already exist")
        return
    
    # load model
    language_model = load_llm(llm_model_name, qlora=True, force_download=True) # quantized model
    llm_param_count = sum([p.numel() for p in language_model.parameters()])
    tokenizer = load_tokenizer(llm_model_name)
    
    tokens = tokenizer(texts, padding="longest", return_tensors="pt")        
    llm_feats, losses, bpb_losses = [], [], []
    
    # hack to get around HF mapping data incorrectly when using model-parallel
    device = next(language_model.parameters()).device
    
    for i in trange(0, len(texts), args.batch_size):
        # get embeddings
        token_inputs = {k: v[i:i+args.batch_size].to(device).long() for (k, v) in tokens.items()}
        
        with torch.no_grad():
            if "olmo" in llm_model_name.lower():
                llm_output = language_model(
                    input_ids=token_inputs["input_ids"],
                    attention_mask=token_inputs["attention_mask"],
                    output_hidden_states=True,
                )
            else:
                llm_output = language_model(
                    input_ids=token_inputs["input_ids"],
                    attention_mask=token_inputs["attention_mask"],
                )
            
            loss, avg_loss = cross_entropy_loss(token_inputs, llm_output)
            losses.extend(avg_loss.cpu())
            
            bpb = cross_entropy_to_bits_per_unit(loss.cpu(), texts[i:i+args.batch_size], unit="byte")
            bpb_losses.extend(bpb)
            
            # make sure to do all the processing in cpu to avoid memory problems
            if args.pooling=='mean':
                feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3)
                mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)
                feats = (feats * mask).sum(2) / mask.sum(2)
            elif args.pooling=='last':
                feats = [v[:, -1, :] for v in llm_output["hidden_states"]]
                feats = torch.stack(feats).permute(1, 0, 2) 
            else:
                raise NotImplementedError(f"unknown pooling {args.pool}")
            # llm_feats.append(feats.cpu())
                        
            # torch.save({"feats": feats.cpu(), 
            #             "num_params": llm_param_count}, save_path.replace(".pt", f"_{i}.pt"))
        
    print(f"average loss:\t{torch.stack(losses).mean().item()}")
    save_dict = {
        "feats": torch.cat(llm_feats).cpu(),
        "num_params": llm_param_count,
        "mask": tokens["attention_mask"].cpu(),
        "loss": torch.stack(losses).mean(),
        "bpb": torch.stack(bpb_losses).mean(),
    }

    torch.save(save_dict, save_path)

    del language_model, tokenizer, llm_feats, llm_output
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
            

if __name__=='__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--modality', type=str, required=True)
    parser.add_argument('--model_family', type=str, required=True)
    parser.add_argument('--pooling', type=str, default='mean')
    
    args = parser.parse_args()
    
    if args.modality == 'vision':
        f_stim = h5py.File('/SSD/slava/THESIS/nsd_stimuli.hdf5', 'r')
        stim = f_stim['imgBrick'][:]
        del f_stim
    elif args.modality == 'language':
        # load captions
        coco_df = pd.read_csv('/SSD/slava/THESIS/total_coco.csv')
        nsd_df = pd.read_pickle('/SSD/slava/brain_decoding/nsd/data/nsddata/experiments/nsd/nsd_stim_info_merged.pkl')
        
        stim = []
        for i, row in nsd_df.iterrows():
            coco_id = row['cocoId']
            caps = coco_df[coco_df['id']==coco_id]['captions']
            caps = ast.literal_eval(caps.iloc[0])
            
            stim.append(', '.join(caps))
        
        print("Length of stimuli: ", len(stim))
        del coco_df, nsd_df
    
    model_names = utils.get_models(args)
    
    if args.modality=='vision':
        for lvm_model in model_names:  
            extract_vision_features(lvm_model, args, stim)
    elif args.modality == 'language':
        for llm_model in model_names:
            extract_llm_features(llm_model, args, stim)
    