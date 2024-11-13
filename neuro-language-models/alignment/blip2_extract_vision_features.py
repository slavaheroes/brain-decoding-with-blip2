import argparse
import os
os.environ['HF_HOME'] = '/home/guest/qasymjomart/LLM/modelhouse/'

from PIL import Image
import h5py
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from tqdm import trange

@torch.no_grad()
def extract_features(lvm_model, processor, images, config, args, savepath):
    idx = 0
    vision_model = lvm_model.vision_model
    query_tokens_ = lvm_model.query_tokens
    qformer = lvm_model.qformer
    
    vision_model.to(args.gpu)
    query_tokens_.to(args.gpu)
    qformer.to(args.gpu)
    
    vision_model.eval()
    # query_tokens_.eval()
    qformer.eval()
    
    del lvm_model
    
    for i in trange(0, len(images), args.batch_size):
        inputs = processor(images=[Image.fromarray(im) for im in images[i:i+args.batch_size]],
                                            return_tensors='pt')['pixel_values']
        inputs = inputs.to(args.gpu)
        
        image_embeds = vision_model(inputs,
                                    return_dict=True,
                                    interpolate_pos_encoding=False,
                                    ).last_hidden_state
        
        image_attention_mask = torch.ones(image_embeds.size()[:-1], 
                                          dtype=torch.long, 
                                          device=image_embeds.device)
        
        query_tokens = query_tokens_.expand(image_embeds.shape[0], -1, -1).to(args.gpu)
        
        query_outputs = qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        hidden_state = query_outputs.last_hidden_state
        print("[hidden_state]", hidden_state.shape)
            
        for b in range(len(hidden_state)):
            feats = hidden_state[b].detach().cpu()
            torch.save(feats, os.path.join(savepath, f'nsd-{idx}.pt'))
            idx += 1
    

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embed_type', type=str, default='vision')
    args = parser.parse_args()
    
    model_names = ["Salesforce/blip2-opt-2.7b"] #, 'xtuner/llava-llama-3-8b-v1_1-transformers']
    
    f_stim = h5py.File('/SSD2/guest/slava/THESIS/nsd_stimuli.hdf5', 'r')
    images = f_stim['imgBrick'][:]
    del f_stim
    
    for model_name in model_names:
        processor = Blip2Processor.from_pretrained(model_name,
                                           cache_dir="/home/guest/qasymjomart/LLM/modelhouse/")
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir="/home/guest/qasymjomart/LLM/modelhouse/"
        )
        config = model.config
           
        savepath = f'/SSD2/guest/slava/THESIS/nlm_vision_features/{model_name}/{args.embed_type}/'
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        print("Saving to", savepath)
        extract_features(model, processor, images, config, args, savepath)        