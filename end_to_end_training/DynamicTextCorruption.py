import torch
import random
from torch.utils.data import Dataset
from datasets import load_dataset

class DynamicSoftMaskDataset(Dataset):
    def __init__(self, dataset_name="sentence-transformers/agnews", text_field="description", 
                 max_len=128, error_rate=0.15, homophone_ratio=0.8, tokenizer=None, map_path="models/token_edit_map.pt"):
        print(f"Loading raw dataset: {dataset_name}...")
        self.raw_data = load_dataset(dataset_name, split="train")[text_field]
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self.error_rate = error_rate
        self.homophone_ratio = homophone_ratio 
        self.vocab_size = tokenizer.vocab_size
        
        # Load the pre-computed map into RAM
        print("Loading Token Edit-Distance Map...")
        self.edit_map = torch.load(map_path)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        clean_text = str(self.raw_data[idx])
        
        encoded = self.tokenizer(
            clean_text, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt"
        )
        target_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        input_ids = target_ids.clone()
        detection_mask = torch.zeros_like(target_ids, dtype=torch.float)
        
        valid_indices = torch.where(
            (target_ids != self.tokenizer.pad_token_id) &
            (target_ids != self.tokenizer.cls_token_id) &
            (target_ids != self.tokenizer.sep_token_id)
        )[0]
        
        num_errors = int(len(valid_indices) * self.error_rate)
        
        if num_errors > 0:
            error_indices = valid_indices[torch.randperm(len(valid_indices))[:num_errors]]
            
            for err_idx in error_indices:
                orig_token_id = target_ids[err_idx].item()
                
                # 80% chance to try for an edit-distance/typo replacement
                if random.random() < self.homophone_ratio and orig_token_id in self.edit_map:
                    # Randomly select one of the valid typo tokens
                    replacement_id = random.choice(self.edit_map[orig_token_id])
                else:
                    # Fallback to random vocabulary token
                    replacement_id = random.randint(1000, self.vocab_size - 1)
                
                input_ids[err_idx] = replacement_id
                detection_mask[err_idx] = 1.0
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'correction_labels': target_ids,
            'detection_labels': detection_mask
        }