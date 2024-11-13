import pandas as pd
import numpy as np

import argparse
import os

import torch 
from torch.cuda.amp import autocast

os.environ['HF_HOME'] = '/home/guest/qasymjomart/LLM/modelhouse/'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import glob

import pytorch_lightning as pl
from transformers import AutoConfig, AddedToken, AutoProcessor, Blip2Processor, Blip2ForImageTextRetrieval
from peft import LoraConfig, get_peft_model
import nlm_modeling as nlm
import utils
import data_utils
from scipy import stats

from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import evaluate
from pycocoevalcap.cider.cider import Cider
from sentence_transformers import SentenceTransformer, util
from transformers import CLIPModel, AutoTokenizer

DEVICE = 'cuda:0'

def load_model(args):
    config['ALIGNMENT_TRAINING']['save_path'] = f"{config['ALIGNMENT_TRAINING']['save_path']}_mem_{args.memory_size}"
    ridge_models = sorted(glob.glob(
        f"{config['ALIGNMENT_TRAINING']['save_path']}/model_*.pkl" 
    ), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    assert len(ridge_models)==32, "Some models are missing"
    brain_model = nlm.BrainToPrefix(ridge_models)
    
    # load NLM
    blip_config = AutoConfig.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        cache_dir="/home/guest/qasymjomart/LLM/modelhouse/"
    )
    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b",
                                            cache_dir="/home/guest/qasymjomart/LLM/modelhouse/")

    # blip_processor.num_query_tokens = blip_config.num_query_tokens
    # image_token = AddedToken("<image>", normalized=False, special=True)
    # blip_processor.tokenizer.add_tokens([image_token], special_tokens=True)
    
    nlm_model = nlm.BrainLanguageModel(
        config=blip_config,
        brain_model=brain_model
    ).from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        cache_dir="/home/guest/qasymjomart/LLM/modelhouse/",
        brain_model=brain_model
    )
    
    # nlm_model.resize_token_embeddings(len(blip_processor.tokenizer), pad_to_multiple_of=64) # pad for efficient computation
    # nlm_model.config.image_token_index = len(blip_processor.tokenizer) - 1
    
    logger.info(f'Loading Model Type: {args.type}')
    
    if args.type in [0]:    
        return (nlm_model, blip_processor), nlm_model.vision_model, None
    elif args.type in [1]:
        # load retrieval model
        blip2_retrieval_model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g", torch_dtype=torch.float32,
                                                   cache_dir="/home/guest/qasymjomart/LLM/modelhouse/")
        blip2_retrieval_processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g",
                                          cache_dir="/home/guest/qasymjomart/LLM/modelhouse/")
        
        retrieval_model = nlm.BrainTextRetrieval(
            brain_model,
            query_tokens=blip2_retrieval_model.query_tokens,
            embeddings=blip2_retrieval_model.embeddings,
            qformer=blip2_retrieval_model.qformer,
            vision_projection=blip2_retrieval_model.vision_projection,
            language_projection=blip2_retrieval_model.text_projection,
            config=blip2_retrieval_model.config    
        )
        
        ckpt = torch.load(args.model_path, map_location='cpu')['state_dict']
        ckpt = utils.format_pl_state_dict(ckpt)
        
        retrieval_model.load_state_dict(ckpt, strict=True)
        nlm_model.vision_model = retrieval_model.brain_model # update vision model
        logger.success('Model Loaded Successfully')
        return (nlm_model, blip_processor), nlm_model.vision_model, (retrieval_model, blip2_retrieval_processor)
    
    
def image_retrieval(args, brain_model, X_test, Y_test):
    percent_correct_fwds, percent_correct_bwds = [], []
    percent_correct_fwd, percent_correct_bwd = None, None
    brain_model.eval()
    brain_model.to(DEVICE)
    
    unique_nsd_ids = defaultdict(list)
    
    for i, y_path in enumerate(Y_test):
        nsd_id = y_path.split('/')[-1].split('.')[0].split('-')[-1]
        nsd_id = int(nsd_id)
        unique_nsd_ids[nsd_id].append(i)
    
    keys = list(unique_nsd_ids.keys())
    ds = data_utils.fMRI_Linear(
        X_test,
        Y_test,
        args.subj,
        split='test',
        memory_size=args.memory_size,
        roi_name=args.roi_name
    )
    
    for test_i, loop in enumerate(tqdm(range(30))):
        random_samps = np.random.choice(np.arange(len(keys)), size=300, replace=False)
        
        brain_embeds = []
        y_embeds = []
        for i in random_samps:
            if len(unique_nsd_ids[keys[i]])>1:
                idx = np.random.choice(unique_nsd_ids[keys[i]], size=1, replace=False)[0]
            else:
                idx = unique_nsd_ids[keys[i]][0]
            
            x, y = ds[idx]
            x = torch.tensor(x).to(DEVICE).unsqueeze(0).float()
            with torch.no_grad():
                brain_embeds.append(
                    brain_model(x).detach().cpu().squeeze(0)
                )
            y_embeds.append(y)
        
        brain_embeds = torch.stack(brain_embeds)
        y_embeds = torch.stack(y_embeds)
        
        if test_i==0:
            logger.info(f'Brain Embeds Shape: {brain_embeds.shape}, Y Embeds Shape: {y_embeds.shape}')
        
        # flatten if necessary
        y_embeds = y_embeds.reshape(len(y_embeds),-1)
        brain_embeds = brain_embeds.reshape(len(brain_embeds),-1)

        # l2norm 
        y_embeds = torch.nn.functional.normalize(y_embeds,dim=-1)
        brain_embeds = torch.nn.functional.normalize(brain_embeds,dim=-1)

        labels = torch.arange(len(y_embeds))
        
        bwd_sim = utils.batchwise_cosine_similarity(y_embeds, brain_embeds)
        fwd_sim = utils.batchwise_cosine_similarity(brain_embeds, y_embeds)
        
        assert len(bwd_sim) == 300
        
        percent_correct_fwds = np.append(percent_correct_fwds, utils.topk(fwd_sim, labels,k=1).item())
        percent_correct_bwds = np.append(percent_correct_bwds, utils.topk(bwd_sim, labels,k=1).item())
        
        if test_i==0:
            print("Loop 0:",percent_correct_fwds, percent_correct_bwds)
                
    percent_correct_fwd = np.mean(percent_correct_fwds)
    fwd_sd = np.std(percent_correct_fwds) / np.sqrt(len(percent_correct_fwds))
    fwd_ci = stats.norm.interval(0.95, loc=percent_correct_fwd, scale=fwd_sd)

    percent_correct_bwd = np.mean(percent_correct_bwds)
    bwd_sd = np.std(percent_correct_bwds) / np.sqrt(len(percent_correct_bwds))
    bwd_ci = stats.norm.interval(0.95, loc=percent_correct_bwd, scale=bwd_sd)

    logger.info(f"fwd percent_correct: {percent_correct_fwd:.4f} 95% CI: [{fwd_ci[0]:.4f},{fwd_ci[1]:.4f}]")
    logger.info(f"bwd percent_correct: {percent_correct_bwd:.4f} 95% CI: [{bwd_ci[0]:.4f},{bwd_ci[1]:.4f}]")
      

def text_retrieval(args, retrieval_model, processor, X_test, Y_test, test_df):
    
    test_df = test_df[test_df['question_type'] == 'caption']
    test_df = test_df[test_df['question'] == '<\s>']
    
    ds = data_utils.fMRI_Text_Retrieval(
        X_test,
        Y_test,
        test_df,
        processor,
        subj=args.subj,
        split='test',
        memory_size=args.memory_size,
        roi_name=args.roi_name
    )
    
    logger.info(f'Len of ds: {len(ds)}')
    
    unique_nsd_ids = defaultdict(list)
    for i, item in enumerate(ds.data):
        _, _, _, nsd_id = item
        unique_nsd_ids[nsd_id].append(i)
        
    keys = list(unique_nsd_ids.keys())
    logger.info(f'Len of keys: {len(keys)}')
    
    percent_correct_img_to_brain, percent_correct_brain_to_img = [], []
    percent_correct_text_to_brain, percent_correct_brain_to_text = [], []
    
    retrieval_model.eval()
    retrieval_model = retrieval_model.to(DEVICE)
    
    for test_i, loop in enumerate(tqdm(range(50))):
        random_samps = np.random.choice(np.arange(len(keys)), size=300, replace=False)
        
        img_embeds = []
        brain_embeds = []
        text_embeds = []
        
        for i in random_samps:
            if len(unique_nsd_ids[keys[i]])>1:
                idx = np.random.choice(unique_nsd_ids[keys[i]], size=1, replace=False)[0]
            else:
                idx = unique_nsd_ids[keys[i]][0]
            
            out = ds[idx]
            
            with torch.no_grad():
                brain_feats, image_feats, text_feats, _ = retrieval_model(
                    torch.tensor(out['fmri']).unsqueeze(0).to(DEVICE).float(),
                    out['vision_embed'].unsqueeze(0).to(DEVICE).float(),
                    out['input_ids'].unsqueeze(0).to(DEVICE).long(),
                    torch.tensor(out['coco_id']).unsqueeze(0).to(DEVICE).long(),
                    out['attention_mask'].unsqueeze(0).to(DEVICE).long(),
                    compute_loss=False
                )
                
            img_embeds.append(image_feats.detach().cpu().squeeze(0))
            brain_embeds.append(brain_feats.detach().cpu().squeeze(0))
            text_embeds.append(text_feats.detach().cpu().squeeze(0))
            
                
        img_embeds = torch.stack(img_embeds)
        brain_embeds = torch.stack(brain_embeds)
        text_embeds = torch.stack(text_embeds)
        
        if test_i==0:
            logger.info(f'Image Embeds Shape: {img_embeds.shape}, Brain Embeds Shape: {brain_embeds.shape}, Text Embeds Shape: {text_embeds.shape}')
        
        # IMAGE & BRAIN
        # flatten if necessary
        img_embeds_ = img_embeds.reshape(len(img_embeds),-1)
        brain_embeds_ = brain_embeds.reshape(len(brain_embeds),-1)
        labels = torch.arange(len(img_embeds_))
        
        bwd_sim = utils.batchwise_cosine_similarity(img_embeds_, brain_embeds_)
        fwd_sim = utils.batchwise_cosine_similarity(brain_embeds_, img_embeds_)
        
        percent_correct_img_to_brain = np.append(percent_correct_img_to_brain, utils.topk(fwd_sim, labels,k=1).item())
        percent_correct_brain_to_img = np.append(percent_correct_brain_to_img, utils.topk(bwd_sim, labels,k=1).item())
        
        # TEXT & BRAIN
        
        # cosine similarity as logits
        logits_per_brain = torch.matmul(
            brain_embeds,
            text_embeds.t()
        )
        logits_per_brain, _ = torch.max(logits_per_brain, dim=1)
        logits_per_text = logits_per_brain.t()
        
        if test_i==0:
            logger.info(f'Logits Per Brain Shape: {logits_per_brain.shape}, Logits Per Text Shape: {logits_per_text.shape}')
        
        percent_correct_text_to_brain = np.append(percent_correct_text_to_brain, utils.topk(logits_per_brain, labels,k=1).item())
        percent_correct_brain_to_text = np.append(percent_correct_brain_to_text, utils.topk(logits_per_text, labels,k=1).item())
        
    
    percent_correct_img_to_brain_ = np.mean(percent_correct_img_to_brain)
    percent_correct_brain_to_img_ = np.mean(percent_correct_brain_to_img)
    percent_correct_text_to_brain_ = np.mean(percent_correct_text_to_brain)
    percent_correct_brain_to_text_ = np.mean(percent_correct_brain_to_text)
    
    logger.info(f"Retrieve image from brain: {percent_correct_img_to_brain_:.4f}")
    logger.info(f"Retrieve brain from image: {percent_correct_brain_to_img_:.4f}")
    
    logger.info(f"Retrieve text from brain: {percent_correct_text_to_brain_:.4f}")
    logger.info(f"Retrieve brain from text: {percent_correct_brain_to_text_:.4f}")


def fmri_captioning(args, nlm_model, processor, X_test, Y_test, test_df):
    test_df = test_df.dropna(subset=['answer'])
    test_df = test_df.dropna(subset=['question'])
    test_df = test_df[test_df['question_type'] == 'caption']
    test_df = test_df[test_df['question'] != '<\s>'] # a photo of prompts
    
    ds = data_utils.fMRI_VQA(
        X_test,
        Y_test,
        test_df,
        processor,
        subj=args.subj,
        split='test',
        memory_size=args.memory_size,
        roi_name=args.roi_name,
        inference=True
    )
    
    logger.info(f'Len of ds: {len(ds)}')
    
    unique_nsd_ids = defaultdict(list)
    nsd_id_to_answers = defaultdict()
    for i, item in enumerate(ds.data):
        nsd_id = item[-1]
        unique_nsd_ids[nsd_id].append(i)
        
        nsd_id_to_answers[nsd_id] = test_df[test_df['nsd_id']==nsd_id]['answer'].to_list()
        
    keys = list(unique_nsd_ids.keys())
    logger.info(f'Len of keys: {len(keys)} | Len of nsd_id_to_answers: {len(nsd_id_to_answers)}')
    
    torch.set_flush_denormal(True)
    torch.cuda.empty_cache()
    
    nlm_model.eval()
    nlm_model.to(DEVICE)
    generated_outputs = defaultdict()
    
    for test_i, loop in enumerate(tqdm(range(2*args.memory_size))):
        
        pred_captions = []
        
        for nsd_id, idxs in tqdm(unique_nsd_ids.items()):
            if len(idxs)>1:
                idx = np.random.choice(idxs, size=1, replace=False)[0]
                if len(idxs)>1:
                    # remove this idx
                    unique_nsd_ids[nsd_id] = [i for i in idxs if i!=idx]
            else:
                idx = idxs[0]
            
            out = ds[idx]
            
            with torch.no_grad(), autocast(dtype=torch.float16):
                output = nlm_model.generate(
                    pixel_values=torch.tensor(out['fmri']).unsqueeze(0).to(DEVICE).float(),
                    input_ids=out['input_ids'].unsqueeze(0).to(DEVICE).long() if test_i%2==0 else None,
                    attention_mask=out['attention_mask'].unsqueeze(0).to(DEVICE).long() if test_i%2==0 else None,
                    # max_new_tokens=50,
                )
            generated_text = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
        
            pred_captions.append(
                {'nsd_id': nsd_id, 'caption': generated_text, 'true_captions': nsd_id_to_answers[nsd_id]}
            )
            
            # print(f"Question: ", ds.data[idx][2] if test_i%2==0 else None)
            if len(pred_captions)%100==0:
                logger.info(f'Question: {processor.batch_decode(out["input_ids"].unsqueeze(0), skip_special_tokens=True)}')
                logger.info(f"Generated Text: {generated_text}")
                logger.info(f"True Text: {nsd_id_to_answers[nsd_id]}")
                # print('-'*50)
            
            # if len(pred_captions)==20: break
            
        # logger.info(f"Generated Text: {generated_text}")
        # logger.info(f"True Text: {nsd_id_to_answers[nsd_id]}")
        
        # assert len(pred_captions)==len(nsd_id_to_answers)
        generated_outputs[test_i] = pd.DataFrame(pred_captions)
    
    best_df, best_clip_L_score = None, 0.0
    best_key = None
    scores = defaultdict()
    
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    cider_scorer = Cider()
    
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sentence_model.to(DEVICE)
    
    model_clip_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                                cache_dir='/home/guest/qasymjomart/LLM/modelhouse/')
    model_clip_base.to(DEVICE)
    model_clip_base.eval()
    tokenizer_base =  AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32",
                                                    cache_dir='/home/guest/qasymjomart/LLM/modelhouse/')
    
    model_clip_large = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",
                                                 cache_dir='/home/guest/qasymjomart/LLM/modelhouse/')
    model_clip_large.to(DEVICE)
    model_clip_large.eval()
    tokenizer_large =  AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14",
                                                     cache_dir='/home/guest/qasymjomart/LLM/modelhouse/')

    for key, df in generated_outputs.items():
        logger.info(f'Start Evaluation for Key: {key}')
        
        nsd_ids = df['nsd_id'].tolist()
        references = df['true_captions'].to_list()
        hypotheses = df['caption'].to_list()
        
        # Meteor
        meteor_score = meteor.compute(references=references, predictions=hypotheses)['meteor']
        logger.info(f'Meteor: {meteor_score}')
        # Rouge
        rouge_1_score = rouge.compute(references=references, predictions=hypotheses)['rouge1']
        rouge_L_score = rouge.compute(references=references, predictions=hypotheses)['rougeL']
        
        logger.info(f'Rouge-1: {rouge_1_score}')
        logger.info(f'Rouge-L: {rouge_L_score}')
        
        # Cider
        cider_input = {str(id): [cap] for id, cap in zip(nsd_ids, hypotheses)}
        cider_refs = {str(id): refs for id, refs in zip(nsd_ids, references)}
        cider_scores, _ = cider_scorer.compute_score(cider_refs, cider_input)
        
        logger.info(f'Cider: {np.mean(cider_scores)}')
        
        # Bleu
        bleu_score_1 = bleu.compute(references=references, predictions=hypotheses, max_order=1)['bleu']
        bleu_score_2 = bleu.compute(references=references, predictions=hypotheses, max_order=2)['bleu']
        bleu_score_3 = bleu.compute(references=references, predictions=hypotheses, max_order=3)['bleu']
        bleu_score_4 = bleu.compute(references=references, predictions=hypotheses, max_order=4)['bleu']
        
        logger.info(f'Bleu-1: {bleu_score_1}')
        logger.info(f'Bleu-2: {bleu_score_2}')
        logger.info(f'Bleu-3: {bleu_score_3}')
        logger.info(f'Bleu-4: {bleu_score_4}')
        
        # Sentence Transformers
        with torch.no_grad():
            embeddings_gt = []
            for i in range(len(references)):
                embeddings_gt.append(
                    sentence_model.encode(references[i], convert_to_tensor=True).mean(dim=0).cpu()
                )
            embeddings_gt = torch.stack(embeddings_gt).cpu()
            embeddings_pred = sentence_model.encode(hypotheses, convert_to_tensor=True).cpu()
            sentence_transformers_score = util.pytorch_cos_sim(embeddings_pred, embeddings_gt).diag().mean().item()
        logger.info(f'Sentence Transformers: {sentence_transformers_score}')
        
        # CLIP-B
        with torch.no_grad():
            embeddings_gt = []
            for i in range(len(references)):
                embeddings_gt.append(
                    model_clip_base.get_text_features(
                        **tokenizer_base(references[i], return_tensors='pt', padding=True, truncation=True).to(DEVICE)
                        ).mean(dim=0).cpu()
                )
            embeddings_gt = torch.stack(embeddings_gt)
            embeddings_pred = model_clip_base.get_text_features(
                **tokenizer_base(hypotheses, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            ).cpu()
            
            clip_base_score = util.pytorch_cos_sim(embeddings_pred, embeddings_gt).diag().mean().item()
        
        logger.info(f'CLIP Base: {clip_base_score}')
        
        # CLIP-L
        with torch.no_grad():
            embeddings_gt = []
            for i in range(len(references)):
                embeddings_gt.append(
                    model_clip_large.get_text_features(
                        **tokenizer_large(references[i], return_tensors='pt', padding=True, truncation=True).to(DEVICE)
                        ).mean(dim=0).cpu()
                )
            embeddings_gt = torch.stack(embeddings_gt)
            embeddings_pred = model_clip_large.get_text_features(
                **tokenizer_large(hypotheses, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            ).cpu()
            
            clip_large_score = util.pytorch_cos_sim(embeddings_pred, embeddings_gt).diag().mean().item()
        
        logger.info(f'CLIP Large: {clip_large_score}')
        print()
        
        scores[key] = {
            'meteor': meteor_score,
            'rouge_1': rouge_1_score,
            'rouge_L': rouge_L_score,
            'cider': np.mean(cider_scores),
            'bleu_1': bleu_score_1,
            'bleu_2': bleu_score_2,
            'bleu_3': bleu_score_3,
            'bleu_4': bleu_score_4,
            'sentence_transformers': sentence_transformers_score,
            'clip_base': clip_base_score,
            'clip_large': clip_large_score
        }
        
        if clip_large_score>best_clip_L_score:
            best_clip_L_score = clip_large_score
            best_df = df
            best_key = key
    
    logger.info(f'Best CLIP Large Score: {best_clip_L_score}')
    logger.info(f'Best Key: {best_key}')
    logger.info(f'Best scores: {scores[best_key]}')
    
    del model_clip_base, model_clip_large, sentence_model
    
    return best_df, scores


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Inference for language models')
    parser.add_argument('--type', type=int, help='Type of model')
    ### 
    # type 0 -> stage 1.1 (train): Image Retrieval, Captioning, VQA
    # type 1 -> stage 1.1 + stage 1.2 (train): Image Retrieval, Text Retrieval, Captioning, VQA
    ###
    parser.add_argument('--roi_name', type=str, help='Name of the region of interest in the brain')
    parser.add_argument('--memory_size', type=int, help='Size of the memory')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--config', type=str, help='Path to the config')
    parser.add_argument('--savename', type=str, help='Name of the file to save the results')
    parser.add_argument('--subj', type=int, help='Subject number')
    args = parser.parse_args()
    
    config = utils.read_yaml(args.config)
    pl.seed_everything(config['SEED'])
    config['DATA']['memory_size'] = args.memory_size
    _, _, X_test, Y_test = utils.load_data(subj=args.subj, config=config)
    test_df = pd.read_csv(
        f'/SSD2/guest/slava/THESIS/VQA/subj{args.subj}_test.csv'
    )
    test_df = test_df.dropna(subset=['answer'])
    test_df = test_df.dropna(subset=['question'])
    
    logger.info(f'Len of X_test: {len(X_test)}, Len of Y_test: {len(Y_test)}')
    logger.info(f'Shape of test_df: {test_df.shape}')
    
    nlm_model, brain_model, text_retrieval_model = load_model(args)
    
    # image retrieval
    
    logger.info('Running Image Retrieval')
    image_retrieval(
        args,
        brain_model,
        X_test,
        Y_test
    )
    
    logger.success('Image Retrieval Done')
    
    # text retrieval
    
    if text_retrieval_model is not None:
        logger.info('Running Text Retrieval')
        text_retrieval(
            args,
            text_retrieval_model[0],
            text_retrieval_model[1],
            X_test,
            Y_test,
            test_df
        )
        logger.success('Text Retrieval Done')
    
    # captioning
    logger.info('Running Captioning')
    pred_df, captioning_results = fmri_captioning(
        args,
        nlm_model[0],
        nlm_model[1],
        X_test,
        Y_test,
        test_df
    )
    
    pred_df.to_csv(f'results/{args.savename}_captions.csv', index=False)
    logger.success('Captioning Done')
    
    logger.info(f'Captioning Results: {captioning_results}')