import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizer
from tqdm import tqdm

# Import your custom modules based on your file structure
from DynamicTextCorruption import DynamicSoftMaskDataset
from CombinedModel import SoftMaskedBertJoint
from BiGRUDetectionNetwork import BiGRUDetectionNetwork
from SoftMaskDistilBert import SoftMaskDistilBert

def train_soft_masked_bert(
    dataset_name="sentence-transformers/agnews",
    text_field="description",
    epochs=1, 
    batch_size=64, 
    lambda_weight=0.8, # From Zhang et al. (0.8 weights correction higher than detection)
    lr=2e-4,
    max_len=128,
    error_rate=0.15,
    homophone_ratio=0.8,
    map_path="models/token_edit_map.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing training on: {device}")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # 1. Init Dynamic Dataset & DataLoader
    # We pass the tokenizer directly to the dataset so it can tokenize and corrupt on the fly
    dataset = DynamicSoftMaskDataset(
        dataset_name=dataset_name,
        text_field=text_field,
        max_len=max_len,
        error_rate=error_rate,
        tokenizer=tokenizer,
        homophone_ratio=0.8,
        map_path="models/token_edit_map.pt"
    )
    
    # Because max_len is fixed, PyTorch's default collate function will easily batch the dictionaries
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # 2. Init Base Models 
    detection_net = BiGRUDetectionNetwork().to(device)
    correction_net = SoftMaskDistilBert().to(device)
    
    # 3. Wrap in Joint Model
    model = SoftMaskedBertJoint(detection_net, correction_net, tokenizer).to(device)
    model.train()
    
    # 4. Optimizers & Loss Functions
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # BCEWithLogits takes raw logits (from Bi-GRU) and applies sigmoid internally for numerical stability
    loss_fn_detection = nn.BCEWithLogitsLoss() 
    # Ignore pad tokens in correction loss so we don't train the model to predict padding
    loss_fn_correction = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) 
    
    print("Starting Joint End-to-End Training...")
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            # Move all dynamic tensors to the GPU/CPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            det_labels = batch['detection_labels'].to(device)
            corr_labels = batch['correction_labels'].to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass through the joint architecture
            det_logits, corr_logits = model(input_ids, attention_mask)
            
            # Calculate Detection Loss (Shape: Batch x SeqLen)
            l_d = loss_fn_detection(det_logits, det_labels)
            
            # Calculate Correction Loss
            # CrossEntropy expects logits of shape (N, C) and labels of shape (N)
            # We flatten the batch and sequence dimensions: (Batch * SeqLen, VocabSize)
            corr_logits_flat = corr_logits.view(-1, corr_logits.size(-1))
            corr_labels_flat = corr_labels.view(-1)
            l_c = loss_fn_correction(corr_logits_flat, corr_labels_flat)
            
            # Combine Losses (Zhang's eq 13)
            total_loss = (lambda_weight * l_c) + ((1.0 - lambda_weight) * l_d)
            
            # Backward pass & Optimizer Step
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}", 
                'l_c': f"{l_c.item():.4f}", 
                'l_d': f"{l_d.item():.4f}"
            })
            
        print(f"Average Loss for Epoch {epoch+1}: {epoch_loss / len(dataloader):.4f}")
        
    # Save the individual model weights
    torch.save(detection_net.state_dict(), "models/bigru_detection_model_trained.pt")
    torch.save(correction_net.state_dict(), "models/softmask_correction_model_trained.pt")
    print("\nTraining Complete. Model weights saved to /models.")

if __name__ == "__main__":
    train_soft_masked_bert()
    
    
