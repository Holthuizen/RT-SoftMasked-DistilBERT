import torch
import torch.nn as nn

class SoftMaskedBertJoint(nn.Module):
    def __init__(self, detection_model, correction_model, tokenizer):
        super().__init__()
        self.detection = detection_model
        self.correction = correction_model
        self.pad_token_id = tokenizer.pad_token_id
        
    def forward(self, input_ids, attention_mask):
        # 1. Detection Pass
        det_logits = self.detection(input_ids, attention_mask)
        error_probs = torch.sigmoid(det_logits)
        
        # Zero-out probabilities for special tokens so we don't mask [CLS] or [SEP]
        is_special = (input_ids == self.pad_token_id) | \
                     (input_ids == self.correction.tokenizer.cls_token_id) | \
                     (input_ids == self.correction.tokenizer.sep_token_id)
        error_probs = torch.where(is_special, torch.tensor(0.0, device=input_ids.device), error_probs)
        
        # 2. Correction Pass
        corr_logits = self.correction(input_ids, attention_mask, error_probs)
        
        return det_logits, corr_logits