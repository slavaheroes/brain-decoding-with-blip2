import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Blip2ForConditionalGeneration
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGenerationModelOutput
from torch.nn import CrossEntropyLoss

from typing import Optional
from loguru import logger
import utils

class BrainToPrefix(nn.Module):
    def __init__(self,
                 path_to_linear_models,
                 ):
        super().__init__()
        self.coefs = torch.nn.ParameterList()
        self.intercepts = torch.nn.ParameterList()
        
        for model_path in path_to_linear_models:
            m = utils.read_pickle(model_path)
            
            self.coefs.append(
                torch.nn.Parameter(torch.tensor(m.coef_, dtype=torch.float32))
            )
            
            self.intercepts.append(
                torch.nn.Parameter(torch.tensor(m.intercept_, dtype=torch.float32))
            )
    
    def forward(self, x):
        out = []
        for i in range(len(self.coefs)):
            out.append(torch.matmul(x, self.coefs[i].T) + self.intercepts[i])
        return torch.stack(out, dim=1)
            

class BrainTextRetrieval(nn.Module):
    def __init__(self,
                 brain_model,
                 query_tokens,
                 embeddings,
                 qformer,
                 vision_projection,
                 language_projection,
                 config,
                 loss_weights=[0.3, 0.7],
                 ):
        super().__init__()
        self.brain_model = brain_model
        self.query_tokens = query_tokens
        self.embeddings = embeddings
        self.qformer = qformer
        self.config = config
        
        self.brain_projection = nn.Linear(config.qformer_config.hidden_size, config.image_text_hidden_size)
        self.vision_projection = vision_projection #nn.Linear(config.qformer_config.hidden_size, config.image_text_hidden_size)
        self.language_projection = language_projection #nn.Linear(config.qformer_config.hidden_size, config.image_text_hidden_size)
        
        self.loss_weights = loss_weights
        self.temp = torch.nn.Parameter(0.07 * torch.ones([]))
        
    def compute_contrastive_loss(self,
                                 feats_1,
                                 feats_2,
                                 coco_ids,
                                 to_squeeze=True,):
        
        # Contrastive Loss
        sim_q2t = torch.matmul(
            feats_1.unsqueeze(1), feats_2.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            feats_2.unsqueeze(1).unsqueeze(1), feats_1.permute(0, 2, 1)
        ).squeeze()
        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]
        
        image_ids = coco_ids.view(-1, 1)
        pos_idx = torch.eq(image_ids, image_ids.T).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        sim_targets = 0.9*sim_targets + 0.1 * torch.ones_like(sim_targets) / sim_targets.size(1)
        
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
        contrastive_loss = (loss_t2i+loss_i2t)/2
        
        return contrastive_loss
        
        
    def forward(self,
                brain_data,
                gt_vision_embed,
                input_ids,
                coco_ids,
                attention_mask: Optional[torch.LongTensor] = None,
                compute_loss=True,
                ):
        
        pred_image_embed = self.brain_model(brain_data)
        
        # MSE Loss
        mse_loss = nn.functional.mse_loss(pred_image_embed, gt_vision_embed)
        # 
                
        query_embeds = self.embeddings(input_ids)
        text_outputs = self.qformer(
            query_embeds=query_embeds,
            query_length=0,
            attention_mask=attention_mask,
            return_dict=True,)
        
        brain_feats = nn.functional.normalize(self.brain_projection(pred_image_embed), dim=-1)
        image_feats = nn.functional.normalize(self.vision_projection(gt_vision_embed), dim=-1)
        text_feats = nn.functional.normalize(self.language_projection(text_outputs.last_hidden_state[:, 0, :]), dim=-1)
        
        if compute_loss:
            brain_text_loss = self.compute_contrastive_loss(brain_feats, text_feats, coco_ids)
            text_image_loss = self.compute_contrastive_loss(image_feats, text_feats, coco_ids)
            contrastive_loss = (brain_text_loss + text_image_loss) / 2
            
            # contrastive_loss = brain_text_loss
            loss = self.loss_weights[0] * contrastive_loss + \
                    self.loss_weights[1] * mse_loss
        
            return brain_feats, image_feats, text_feats, loss, contrastive_loss, mse_loss

        return brain_feats, image_feats, text_feats, mse_loss
        
         
class BrainLanguageModel(Blip2ForConditionalGeneration):
    def __init__(
        self,
        config,
        brain_model
    ):
        super().__init__(config)
        self.vision_model = brain_model
        # self.query_tokens = torch.nn.Identity()
        self.qformer = torch.nn.Identity()
        
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False):
        
        vision_outputs = self.vision_model(pixel_values)
        query_output = vision_outputs
        
        language_model_inputs =  self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # if the model already has "image_token_index" then the input is expanded to account for image embeds
        # otherwise we expand manually by concating
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
        
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
        attention_mask = torch.cat(
            [language_model_attention_mask, attention_mask.to(language_model_attention_mask.device)], dim=1
        )

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)
                # print("labels: ", labels.shape)
                # print('logits: ', logits.shape)
                logits = logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # print(shift_logits.shape, shift_labels.shape)
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="mean")
                # print(shift_logits.view(-1, self.config.text_config.vocab_size).shape, shift_labels.view(-1).shape)
                
                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
                

        if not return_dict:
            output = (logits, vision_outputs, vision_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=vision_outputs,
            language_model_outputs=outputs,
        )
        
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ):
        batch_size = pixel_values.shape[0]
        feat = self.vision_model(pixel_values)
        query_output = feat
    
        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(feat.device)
            )
        inputs_embeds = self.get_input_embeddings()(input_ids)
        
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
        
        if not self.language_model.config.is_encoder_decoder:
            generate_kwargs["max_new_tokens"] = (
                generate_kwargs.get("max_new_tokens", 20) + language_model_inputs.shape[1] - 1
            )
            generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + language_model_inputs.shape[1]
                
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        # this is a temporary workaround to be consistent with other generation models and
        # have BOS as the first token, even though under the hood we are calling LM with embeds
        if not self.language_model.config.is_encoder_decoder:
            bos_tokens = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(feat.device)
            )
            if not isinstance(outputs, torch.Tensor):
                outputs.sequences = torch.cat([bos_tokens, outputs.sequences], dim=-1)
            else:
                outputs = torch.cat([bos_tokens, outputs], dim=-1)
        return outputs