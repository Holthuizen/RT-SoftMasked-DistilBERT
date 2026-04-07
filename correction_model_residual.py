import torch
import torch.nn as nn
from transformers import DistilBertForMaskedLM, DistilBertTokenizer


""" 
NOTE this model has the Residual connection ADDED, 
in the paper by Zhang they use model-R to note a model with its residual removed.
I find that a too confusing naming convention to follow
"""
DEBUG = True

class SoftMaskDistilBert(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.model = DistilBertForMaskedLM.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        #cache masked token for soft-masking
        self.mask_token_id = self.tokenizer.mask_token_id
        
        
    def forward(self, input_ids, attention_mask, error_probs, inputs_embeds=None): 
        
        if inputs_embeds != None:
            #grab input embeddings e'_i and e_mask (assuming that input to the model are the soft-masked embeds E' not E)
            inputs_embeds = self.model.distilbert.embeddings(input_ids)
        #full_like: create identical tensors as input_embeds (same shape and type etc.) but filled with the mask_id values.
        mask_ids = torch.full_like(input_ids, self.mask_token_id)
        mask_embeds = self.model.distilbert.embeddings(mask_ids)
        
        p_i = error_probs.unsqueeze(-1)
        #eq. 5  linear combination of mask and input embeddings     
        soft_masked_embeds = (p_i*mask_embeds) + ((1-p_i) * inputs_embeds)

        #transformer forward pass 
        base_outputs = self.model.distilbert(
            inputs_embeds = soft_masked_embeds,
            attention_mask = attention_mask
        )
        
        if DEBUG: 
            print(base_outputs)
        
        #residual injection eq. 10 but modified to take e' as thats available by the model (and makes sense to me for a residual)
        h_prime = base_outputs.last_hidden_state + inputs_embeds
        
        #unrolling the 768x30k upscaling to insert the modified h_prime state 
        #Linear transform --> GELU --> Layer norm --> projection to vocab
        logits = self.model.vocab_transform(h_prime)
        logits = nn.functional.gelu(logits)
        logits = self.model.vocab_layer_norm(logits)
        logits = self.model.vocab_projector(logits)
        
        return logits