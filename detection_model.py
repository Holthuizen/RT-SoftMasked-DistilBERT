import torch.nn as nn
from transformers import DistilBertModel

#Bi-GRU detection model inspired by work from Zhang at el Spelling Error Correction with Soft-Masked BERT
class BiGRUDetectionNetwork(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', hidden_dim=256):
        super().__init__()
        # Load DistilBERT strictly for its embedding layer
        distilbert = DistilBertModel.from_pretrained(model_name)
        self.embeddings = distilbert.embeddings
        
        # Freeze embeddings so we only train the Bi-GRU weights
        for param in self.embeddings.parameters():
            param.requires_grad = False
            
        embed_dim = distilbert.config.dim
        
        # The Bi-GRU layer
        self.bigru = nn.GRU(
            input_size=embed_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=True
        )
        
        # Project hidden states to a single binary logit
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids, attention_mask):
        embeds = self.embeddings(input_ids)
        gru_out, _ = self.bigru(embeds)
        logits = self.classifier(gru_out).squeeze(-1) # Output shape: (batch_size, seq_len)
        return logits