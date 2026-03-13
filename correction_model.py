import torch
import torch.nn as nn
from transformers import DistilBertForMaskedLM, DistilBertTokenizer

# BERT modification inspired by work from Zhang at el Spelling Error Correction with Soft-Masked BERT

class SoftMaskDistilBert(nn.Module): 
    def __init__(self, model_name="distilbert-base-uncased"):
        #init nn.model class
        super().__init__()
        #load MLM pretrained model and tokenizer
        self.model = DistilBertForMaskedLM.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        #cache masked token id
        self.mask_token_id = self.tokenizer.mask_token_id
        
    def forward(self, input_ids, attention_mask, error_probs): 
        
        input_embeds = self.model.distilbert.embeddings(input_ids)
        
        mask_ids = torch.full_like(input_ids, self.mask_token_id)
        mask_embeds = self.model.distilbert.embeddings(mask_ids)
        
        #SOFTMASKING 
        
        #expand error prob to match embedding dims (batch_size, seq_len, 1)
        p_i = error_probs.unsqueeze(-1)
        
        #softmask eq: e'_1 = p_i *e_mask + (1-p_i) * e_i
        soft_masked_embeds = (p_i * mask_embeds) + ((1-p_i) * input_embeds)
        
        #full pass, bypass token ID lookup by setting input_embeds 
        outputs = self.model(inputs_embeds=soft_masked_embeds, attention_mask=attention_mask)
        
        return outputs.logits
        

        

