import argparse
import os
import pandas as pd
import glob

import torch
torch.set_float32_matmul_precision('high') # NVIDIA RTX A6000

import pytorch_lightning as pl
from transformers import AutoProcessor, Blip2ForImageTextRetrieval
from transformers import get_cosine_schedule_with_warmup

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from loguru import logger

import nlm_modeling as nlm
import data_utils
import utils


class ContrastiveLearner(pl.LightningModule):
    def __init__(self,
                 model,
                 lr,
                 config,
                 train_dataloader,
                 val_dataloader
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'train_dataloader', 'val_dataloader'])
        self.model = model
        self.lr = lr
        self.config = config
        
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        
        self.valitation_outputs = []
    
    def forward(self, batch):
        return self.model(
            brain_data = batch['fmri'].float(),
            gt_vision_embed = batch['vision_embed'],
            coco_ids = batch['coco_id'],
            input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask'] if 'attention_mask' in batch else None,
        )
    
    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
      
    def training_step(self, batch, batch_idx):
        _, _, _, loss, _, _ = self.forward(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        _, _, _, loss, contrastive_loss, mse_loss = self.forward(batch)
        self.valitation_outputs.append({'loss': loss,
                                        'contrastive_loss': contrastive_loss,
                                        'mse_loss': mse_loss})    
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.valitation_outputs]).mean()
        avg_contrastive_loss = torch.stack([x['contrastive_loss'] for x in self.valitation_outputs]).mean()
        avg_mse_loss = torch.stack([x['mse_loss'] for x in self.valitation_outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_contrastive_loss', avg_contrastive_loss)
        self.log('val_mse_loss', avg_mse_loss)
        self.valitation_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([{'params': self.model.brain_model.parameters(), 'lr': self.lr/5},
                                       {'params': self.model.qformer.parameters(), 'lr': self.lr},
                                       {'params': self.model.embeddings.parameters(), 'lr': self.lr},
                                       {'params': self.model.query_tokens, 'lr': self.lr},
                                       {'params': self.model.brain_projection.parameters(), 'lr': self.lr},
                                       {'params': self.model.vision_projection.parameters(), 'lr': self.lr},
                                       {'params': self.model.language_projection.parameters(), 'lr': self.lr},
                                       {'params': self.model.temp, 'lr': self.lr}
                                    ], 
                                      lr=self.lr,
                                      **self.config['CONTRASTIVE_TRAINING']['optimizer_args'])
        
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                    num_warmup_steps=self.config['CONTRASTIVE_TRAINING']['warmup_steps'],
                    num_training_steps=self.config['CONTRASTIVE_TRAINING']['total_steps'])
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--memory_size', type=int, required=True)
    parser.add_argument('--savename', type=str, required=True)
    parser.add_argument('--subj', type=int, required=True)
    args = parser.parse_args()
    
    config = utils.read_yaml(args.config)
    pl.seed_everything(config['SEED'])
    
    wand_logger = WandbLogger(
        entity='slavaheroes',
        project='nlm_contrastive',
        name=args.savename
    )
    
    config['CLI_ARGS'] = args
    config['DATA']['memory_size'] = args.memory_size
    config['ALIGNMENT_TRAINING']['save_path'] = f"{config['ALIGNMENT_TRAINING']['save_path']}_mem_{args.memory_size}"
    X_train, Y_train, X_test, Y_test = utils.load_data(subj=args.subj, config=config)
    
    ridge_models = sorted(glob.glob(
        f"{config['ALIGNMENT_TRAINING']['save_path']}/model_*.pkl" 
    ), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    assert len(ridge_models)==32, "Some models are missing"
    
    logger.info(f'Models loaded: {ridge_models[:4]} ... {ridge_models[-2:]}')
    
    model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g", torch_dtype=torch.float32,
                                                   cache_dir="/home/guest/qasymjomart/LLM/modelhouse/")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g",
                                          cache_dir="/home/guest/qasymjomart/LLM/modelhouse/")
    
    retrieval_model = nlm.BrainTextRetrieval(
        brain_model=nlm.BrainToPrefix(ridge_models),
        query_tokens=model.query_tokens,
        embeddings=model.embeddings,
        qformer=model.qformer,
        vision_projection=model.vision_projection,
        language_projection=model.text_projection,
        config=model.config    
    )
    
    config['CONTRASTIVE_TRAINING']['blip_config'] = model.config
    del model
    
    logger.info("Contrastive Model Initialized")
    
    train_df = pd.read_csv(
        f'/SSD2/guest/slava/THESIS/VQA/subj{args.subj}_train.csv'
    )
    train_df = train_df[train_df['question_type'] == 'caption']
    train_df = train_df[train_df['question'] == '<\s>']
    
    test_df = pd.read_csv(
        f'/SSD2/guest/slava/THESIS/VQA/subj{args.subj}_test.csv'
    )
    test_df = test_df[test_df['question_type'] == 'caption']
    test_df = test_df[test_df['question'] == '<\s>']
    
    train_ds = data_utils.fMRI_Text_Retrieval(
        X_train, Y_train, train_df, processor,
        subj=args.subj, split='train', memory_size=args.memory_size
    )
    
    test_ds = data_utils.fMRI_Text_Retrieval(
        X_test, Y_test, test_df, processor,
        subj=args.subj, split='test', memory_size=args.memory_size
    )
    
    logger.info(f"Data loaded: {len(train_ds)} train samples, {len(test_ds)} test samples")
    

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=config['CONTRASTIVE_TRAINING']['batch_size'], shuffle=True, num_workers=8
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=config['CONTRASTIVE_TRAINING']['batch_size'], shuffle=False, num_workers=8
    )
    
    logger.info(f"Dataloaders created: {len(train_dl)} train batches, {len(test_dl)} test batches")
    
    config['CONTRASTIVE_TRAINING']['total_steps'] = len(train_dl) * config['CONTRASTIVE_TRAINING']['epochs']
    
    config['CONTRASTIVE_TRAINING']['save_path'] = os.path.join(
        config['CONTRASTIVE_TRAINING']['save_path'], f'subj{args.subj}_memory{args.memory_size}'
    )
    
    os.makedirs(config['CONTRASTIVE_TRAINING']['save_path'], exist_ok=True)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, mode='min'),
        ModelCheckpoint(
            dirpath=config['CONTRASTIVE_TRAINING']['save_path'],
            filename=f'{args.savename}_best_model' + '_{epoch}_{val_loss:.4f}',
            monitor='val_loss',
            save_top_k=1,
            mode='min',
            verbose=True
        ),
        LearningRateMonitor(logging_interval='step'),
    ]
    
    pl_model = ContrastiveLearner(retrieval_model, 
                                  lr=config['CONTRASTIVE_TRAINING']['lr'], 
                                  config=config,
                                  train_dataloader=train_dl,
                                  val_dataloader=test_dl)
    
    trainer = pl.Trainer(
        logger=wand_logger,
        callbacks=callbacks,
        max_epochs=config['CONTRASTIVE_TRAINING']['epochs'],
        devices=[1],
        accelerator='gpu',
        accumulate_grad_batches=2,
        val_check_interval=0.20,
        gradient_clip_val=1.0,
    )
    
    lr_finder = trainer.tuner.lr_find(pl_model, train_dl, test_dl)
    
    pl_model.lr = lr_finder.suggestion()
    logger.info(f"Learning rate suggestion: {pl_model.lr}")
    
    trainer.validate(pl_model)
     
    trainer.fit(pl_model)
    trainer.validate(pl_model, test_dl, ckpt_path='best', verbose=True)
    
    logger.success("Contrastive Training Finished")