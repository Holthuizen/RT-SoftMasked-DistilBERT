import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertLayer

#hierarchy:
#BertModel--> BertEncoder --> BertLayer --> (BertAttention, BertIntermediate, BertOutputs)

DEBUG = True

config = BertConfig(output_hidden_states=True)
base_model = BertModel(config)

if DEBUG:
    print(config.hidden_size, config.max_position_embeddings)


class CustomBert(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.bert = base_model
        
        """
            note: check necessity of this transform
            learnable projection. this is not strictly needed as the dims of embeddings and the out layer already matches. 
            The question is if the soft_masked paper by zhang at el. preforms this projection or not
        """
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        
        #init as identity matrix in place https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.eye_
        nn.init.eye_(self.transform.weight)
        #init bias to zero
        nn.init.zeros_(self.transform.bias)
        
        
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        """with output_hidden_states=True, hidden_states should be a tuple of all 13 layers"""
        embeddings = outputs.hidden_states[0]
        last_layer = outputs.hidden_states[-1] #h_l
        
        if DEBUG: 
            print("Embedding shape", embeddings.shape)
            print("Layer 11 shape", last_layer.shape)
        
        residual = self.transform(embeddings) 
        return last_layer + residual
        
        
