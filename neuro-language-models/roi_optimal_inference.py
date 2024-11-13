import argparse
import os
os.environ['HF_HOME'] = '/home/guest/qasymjomart/LLM/modelhouse/'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import pytorch_lightning as pl
import pandas as pd

from transformers import AutoConfig, Blip2Processor, AutoProcessor, Blip2ForImageTextRetrieval

import glob
from typing import Optional

import nlm_modeling as nlm
import data_utils
import utils

def roi_specific_generate(
    nlm_model,
    roi,
    input_ids,
    attention_mask: Optional[torch.LongTensor] = None,
    interpolate_pos_encoding: bool = False,
    **generate_kwargs,
):
    batch_size = 1
    query_output = roi
    
    language_model_inputs = nlm_model.language_projection(query_output)
    language_attention_mask = torch.ones(
        language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
    )
    
    if input_ids is None:
        input_ids = (
            torch.LongTensor([[nlm_model.config.text_config.bos_token_id]])
            .repeat(batch_size, 1)
            .to(roi.device)
        )
    inputs_embeds = nlm_model.get_input_embeddings()(input_ids)
    
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    
    # if the model already has "image_token_index" then the input is expanded to account for image embeds
    # otherwise we expand manually by concatenating
    # if getattr(self.config, "image_token_index", None) is not None:
    #     special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
    #     language_model_inputs = language_model_inputs.to(inputs_embeds.device, inputs_embeds.dtype)
    #     inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, language_model_inputs)
    # else:
        # logger.warning(
        #     "Expanding inputs for image tokens in BLIP-2 should be done in processing. "
        #     "Please follow instruction here (https://gist.github.com/zucchini-nlp/e9f20b054fa322f84ac9311d9ab67042) to update your BLIP-2 model. "
        #     "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
        # )
        # print('Before: ', language_model_inputs.shape, inputs_embeds.shape)
    inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
    # print('After: ', inputs_embeds.shape)
    attention_mask = torch.cat(
        [language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1
    )
    
    if not nlm_model.language_model.config.is_encoder_decoder:
        generate_kwargs["max_new_tokens"] = (
            generate_kwargs.get("max_new_tokens", 20) + language_model_inputs.shape[1] - 1
        )
        generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + language_model_inputs.shape[1]
            
    outputs = nlm_model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        **generate_kwargs,
    )

    # this is a temporary workaround to be consistent with other generation models and
    # have BOS as the first token, even though under the hood we are calling LM with embeds
    if not nlm_model.language_model.config.is_encoder_decoder:
        bos_tokens = (
            torch.LongTensor([[nlm_model.config.text_config.bos_token_id]])
            .repeat(batch_size, 1)
            .to(roi.device)
        )
        if not isinstance(outputs, torch.Tensor):
            outputs.sequences = torch.cat([bos_tokens, outputs.sequences], dim=-1)
        else:
            outputs = torch.cat([bos_tokens, outputs], dim=-1)
    return outputs

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory_size', type=int, default=1)
    parser.add_argument('--subj', type=int, default=1)
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    
    config = utils.read_yaml(args.config)
    pl.seed_everything(config['SEED'])
    config['DATA']['memory_size'] = args.memory_size
    
    config['ALIGNMENT_TRAINING']['save_path'] = f"{config['ALIGNMENT_TRAINING']['save_path']}_mem_{args.memory_size}"
    ridge_models = sorted(glob.glob(
        f"{config['ALIGNMENT_TRAINING']['save_path']}/model_*.pkl" 
    ), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    assert len(ridge_models)==32, "Some models are missing"
    
    roi_names = [
        'visual-V1',
        'visual-V2',
        'visual-V3',
        'visual-V4',
        'visual',
        'floc-words',
        'floc-bodies',
        'floc-faces',
        'floc-places',
    ]
    
    ROI_STIMULI_RESULTS = []
    
    type_0_model = nlm.BrainToPrefix(ridge_models)
    blip_config = AutoConfig.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        cache_dir="/home/guest/qasymjomart/LLM/modelhouse/"
    )
    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b",
                                            cache_dir="/home/guest/qasymjomart/LLM/modelhouse/")
    
    nlm_model = nlm.BrainLanguageModel(
        config=blip_config,
        brain_model=type_0_model
    ).from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        cache_dir="/home/guest/qasymjomart/LLM/modelhouse/",
        brain_model=type_0_model
    )
    nlm_model.cuda()
    
    type_1_model = glob.glob(f'/SSD2/guest/slava/THESIS/model_checkpoints/contrastive_models_pt/subj{args.subj}_memory{args.memory_size}/*ckpt')[0]
    blip2_retrieval_model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g", torch_dtype=torch.float32,
                                                   cache_dir="/home/guest/qasymjomart/LLM/modelhouse/")
    blip2_retrieval_processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g",
                                        cache_dir="/home/guest/qasymjomart/LLM/modelhouse/")
    retrieval_model = nlm.BrainTextRetrieval(
        type_0_model,
        query_tokens=blip2_retrieval_model.query_tokens,
        embeddings=blip2_retrieval_model.embeddings,
        qformer=blip2_retrieval_model.qformer,
        vision_projection=blip2_retrieval_model.vision_projection,
        language_projection=blip2_retrieval_model.text_projection,
        config=blip2_retrieval_model.config    
    )
    ckpt = torch.load(type_1_model, map_location='cpu')['state_dict']
    ckpt = utils.format_pl_state_dict(ckpt)
    retrieval_model.load_state_dict(ckpt, strict=True)
    type_1_model = retrieval_model.brain_model
    
    del ckpt, retrieval_model, blip2_retrieval_model, blip2_retrieval_processor
    
    const = 11
    for roi_name in roi_names:
        roi = data_utils.get_roi(args.subj, roi_name)
        roi = torch.tensor(roi).float().unsqueeze(0).cuda()
        
        for model_id, brain_model in enumerate([type_0_model, type_1_model]):
            brain_model = brain_model.cuda()
            brain_model.eval()
            
            pred = []
            with torch.no_grad():
                for coef, intercept in zip(brain_model.coefs, brain_model.intercepts):
                    p = torch.matmul(roi, coef.T)
                    p = p / (torch.norm(p, dim=1, keepdim=True) + 1e-8)
                    p = (p*const) + intercept
                    pred.append(p)

            pred = torch.cat(pred, dim=0).unsqueeze(0)
            
            input_ids = None
            attention_mask = None
            
            for temperature in [0.0]:
                output = roi_specific_generate(nlm_model, pred, input_ids,
                                                max_new_tokens=50,
                                                temperature=temperature,
                                                do_sample=True if temperature>0.0 else False,
                                                )
                text = blip_processor.batch_decode(output, skip_special_tokens=True)[0].strip()
                ROI_STIMULI_RESULTS.append({
                    'roi': roi_name,
                    'temperature': temperature,
                    'const': const,
                    'text': text,
                    'model_type': model_id
                })
                
                print(f"ROI: {roi_name}, Temperature: {temperature}, Const: {const}, Model: {model_id}")
                print(text)
                print()
    
    df = pd.DataFrame(ROI_STIMULI_RESULTS)
    df.to_csv(f"results/roi_specific_results_{args.subj}.csv", index=False)