from torch.utils.data import Dataset
import numpy as np
import torch
import nibabel as nib
from utils import load_fmri


def get_roi(subj, roi_name):
    nsdgeneral_mask = nib.load(f'/SSD2/guest/slava/THESIS/NSD_processed/subj0{subj}/roi/nsdgeneral.nii.gz').get_fdata()


    if 'visual' in roi_name:
        visual_roi = nib.load(f'/SSD2/guest/slava/THESIS/NSD_processed/subj0{subj}/roi/prf-visualrois.nii.gz').get_fdata()
        
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
        roi = nib.load(f'/SSD2/guest/slava/THESIS/NSD_processed/subj0{subj}/roi/floc-words.nii.gz').get_fdata()
    elif roi_name=='floc-faces':
        roi = nib.load(f'/SSD2/guest/slava/THESIS/NSD_processed/subj0{subj}/roi/floc-faces.nii.gz').get_fdata()
    elif roi_name=='floc-places':
        roi = nib.load(f'/SSD2/guest/slava/THESIS/NSD_processed/subj0{subj}/roi/floc-places.nii.gz').get_fdata()
    elif roi_name=='floc-bodies':
        roi = nib.load(f'/SSD2/guest/slava/THESIS/NSD_processed/subj0{subj}/roi/floc-bodies.nii.gz').get_fdata()
    elif roi_name=='general':
        roi = nsdgeneral_mask
    else:
        raise ValueError('Unknown roi_name')

    roi = roi[nsdgeneral_mask>0]
    roi[roi<0] = 0
    roi[roi>0] = 1
    
    return roi


class fMRI_Linear(Dataset):
    def __init__(self, X_paths, Y_paths, subj=1, split='train', memory_size=1, roi_name=None):
        self.x_paths = X_paths
        self.y_paths = Y_paths
        self.mask = nib.load(f'/SSD2/guest/slava/THESIS/NSD_processed/subj0{subj}/roi/nsdgeneral.nii.gz').get_fdata()
        self.split = split
        self.shape = self.mask.shape
        self.memory_size = memory_size
        
        if roi_name:
            self.roi = get_roi(subj, roi_name)
        else:
            roi = np.ones_like(self.mask)
            self.roi = roi[self.mask>0]
        
        self.roi = np.concatenate([self.roi for _ in range(memory_size)])
    
    def __len__(self):
        return len(self.x_paths)
    
    def __getitem__(self, idx):
        if self.memory_size > 1:
            x = load_fmri(self.x_paths[idx], shape=self.shape, roi=self.mask)
            y = torch.load(self.y_paths[idx])
        else:
            x = np.array([ load_fmri([x_path], shape=self.shape, roi=self.mask) 
                          for x_path in self.x_paths[idx]]).mean(0)
            y = torch.load(self.y_paths[idx])
        
        # apply roi_mask
        x[self.roi==0] = 0
        
        return x, y


# def fmri_text_retrieval_collate_fn(batch):
#     fmri = torch.stack([torch.tensor(item["fmri"]) for item in batch])
#     vision_embed = torch.stack([torch.tensor(item["vision_embed"]) for item in batch])
#     coco_id = torch.stack([torch.tensor(item["coco_id"], dtype=torch.long) for item in batch])
    
#     return {"fmri": fmri,
#             "vision_embed": vision_embed,
#             # "gt_caption": gt_caption,
#             "coco_id": coco_id}   

class fMRI_Text_Retrieval(Dataset):
    def __init__(self,
                 X_paths,
                 Y_paths,
                 captions_df,
                 processor,
                 subj=1,
                 split='train',
                 memory_size=1,
                 roi_name=None
                 ):
        self.x_paths = X_paths
        self.y_paths = Y_paths
        self.processor = processor
        assert len(self.x_paths)==len(self.y_paths)
        
        self.captions_df = captions_df
        self.mask = nib.load(f'/SSD2/guest/slava/THESIS/NSD_processed/subj0{subj}/roi/nsdgeneral.nii.gz').get_fdata()
        
        self.split = split
        self.shape = self.mask.shape
        self.memory_size = memory_size
        
        self.data = []
        
        for x, y in zip(self.x_paths, self.y_paths):
            nsd_id = y.split('/')[-1].split('.')[0].split('-')[-1]
            nsd_id = int(nsd_id)
            captions = captions_df[captions_df['nsd_id']==nsd_id]['answer'].values.tolist()
            
            for cap in captions:
                self.data.append((x, y, cap, nsd_id))
        
        if roi_name:
            self.roi = get_roi(subj, roi_name)
        else:
            roi = np.ones_like(self.mask)
            self.roi = roi[self.mask>0]
        
        self.roi = np.concatenate([self.roi for _ in range(memory_size)])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x_paths, y_path, caption, nsd_id = self.data[idx]
        
        if self.memory_size > 1:
            x = load_fmri(x_paths, shape=self.shape, roi=self.mask)
            y = torch.load(y_path)
        else:
            x = np.array([ load_fmri([x_path], shape=self.shape, roi=self.mask) 
                          for x_path in x_paths]).mean(0)
            y = torch.load(y_path)
        
        # apply roi_mask
        x[self.roi==0] = 0
        
        text_inputs = self.processor.tokenizer(caption, 
                                               return_tensors="pt", 
                                               padding="max_length", 
                                               max_length=64, 
                                               truncation=True)
        
        return {"fmri": x,
                "vision_embed": y,
                "coco_id": nsd_id,
                'input_ids': text_inputs['input_ids'].squeeze(),
                'attention_mask': text_inputs['attention_mask'].squeeze()}
        

class fMRI_VQA(Dataset):
    def __init__(self,
                 X_paths,
                 Y_paths,
                 df,
                 processor,
                 subj=1,
                 split='train',
                 memory_size=1,
                 roi_name=None,
                 inference=False) -> None:
        super().__init__()
        
        self.x_paths = X_paths
        self.y_paths = Y_paths
        self.processor = processor
        assert len(self.x_paths)==len(self.y_paths)
        
        self.df = df
        self.mask = nib.load(f'/SSD2/guest/slava/THESIS/NSD_processed/subj0{subj}/roi/nsdgeneral.nii.gz').get_fdata()
        
        if roi_name:
            self.roi = get_roi(subj, roi_name)
        else:
            roi = np.ones_like(self.mask)
            self.roi = roi[self.mask>0]
        
        self.roi = np.concatenate([self.roi for _ in range(memory_size)])
        
        self.split = split
        self.shape = self.mask.shape
        self.memory_size = memory_size
        
        self.data = []
        
        for x, y in zip(self.x_paths, self.y_paths):
            nsd_id = y.split('/')[-1].split('.')[0].split('-')[-1]
            nsd_id = int(nsd_id)
            q_a_pairs = df[df['nsd_id']==nsd_id]
            
            for _, row in q_a_pairs.iterrows():
                if row['question_type']=='vqa':
                    question = "Question: " + row['question'] + " Answer: " # + row['answer']
                elif row['question_type']=='caption':
                    question = row['question'] #+ " " + row['answer']
                                      
                if inference:                    
                    self.data.append((x, y, question, row['answer'], row['question_id'], nsd_id))
                else:
                    self.data.append((x, y, question, row['answer'], nsd_id))
        
        self.inference = inference
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.inference:
            x_paths, y_path, question, answer, _, nsd_id = self.data[idx]
        else:
            x_paths, _, question, answer, nsd_id = self.data[idx]
        
        if self.memory_size > 1:
            x = load_fmri(x_paths, shape=self.shape, roi=self.mask)
        else:
            x = np.array([ load_fmri([x_path], shape=self.shape, roi=self.mask) 
                          for x_path in x_paths]).mean(0)
            
        # apply roi_mask
        x[self.roi==0] = 0
        
        question_encoding = self.processor.tokenizer(question, 
                                            return_tensors="pt", 
                                            padding="max_length", 
                                            max_length=32, 
                                            truncation=True)
        
        if self.inference:
            return {"fmri": x,
                    "nsd_id": nsd_id,
                    'input_ids': question_encoding['input_ids'].squeeze(),
                    'attention_mask': question_encoding['attention_mask'].squeeze()
                    }
        
        answer_encoding = self.processor.tokenizer(answer, 
                                          return_tensors="pt", 
                                          padding="max_length", 
                                          max_length=32, 
                                          truncation=True)
        
        # Combine question and answer for input_ids and attention_mask
        input_ids = torch.cat([question_encoding['input_ids'], answer_encoding['input_ids']], dim=1)
        attention_mask = torch.cat([question_encoding['attention_mask'], answer_encoding['attention_mask']], dim=1)
        # Create labels: -100 for question tokens, actual token ids for answer
        labels = torch.full_like(input_ids, -100)
        labels[:, question_encoding['input_ids'].shape[1]:] = answer_encoding['input_ids']
        
        return {"fmri": x,
                "nsd_id": nsd_id,
                'input_ids': input_ids.squeeze(),
                'attention_mask': attention_mask.squeeze(),
                'labels': labels.squeeze()
                }   
                