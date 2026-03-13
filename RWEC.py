import torch
from correction_model import SoftMaskDistilBert
from detection_model import BiGRUDetectionNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("runtime device: ", device)

detection_model = BiGRUDetectionNetwork().to(device)
detection_model.load_state_dict(torch.load("models/bigru_detection_model.pt"))

correction_model = SoftMaskDistilBert().to(device)
correction_model.eval()

tokenizer = correction_model.tokenizer


def soft_mask_encoding_decoding_pipeline(X, detection_model, correction_model, tokenizer): 
        
    inputs = tokenizer(X, padding=True, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs["attention_mask"] 
    
    with torch.no_grad():
        #bi-GRU model predicts binary masks 0 and 1 
        detection_logits = detection_model(input_ids, attention_mask)
        #confidences
        error_probs = torch.sigmoid(detection_logits)
        
        #boolean mask for special character token ID's that got corrupted via the noisy masking and need to be set to 0
        is_special = (input_ids == tokenizer.pad_token_id) | (input_ids == tokenizer.cls_token_id) | (input_ids == tokenizer.sep_token_id)
        error_probs = torch.where(is_special, torch.tensor(0.0, device=device), error_probs) 

        #BERT based correction network 
        correction_logits = correction_model(input_ids, attention_mask, error_probs)
        predicted_token_ids = torch.argmax(correction_logits, dim=-1)

        #place the special tokens back after the model pass
        predicted_token_ids = torch.where(is_special, input_ids, predicted_token_ids)  
        
    #reverse token-id lookup to turn ids into text
    decoded_texts = tokenizer.batch_decode(predicted_token_ids,skip_special_tokens=True)
    return decoded_texts


# --- Let's test the entire architecture! ---
test_sentences = [
    "loop up that word in the dictionary",
    "Sony will flay EU debut of PlayStation 3",
    "Police Use Taser on Python to Tree Man"
]

final_predictions = soft_mask_encoding_decoding_pipeline(test_sentences, detection_model, correction_model, tokenizer)

for orig, corrected in zip(test_sentences, final_predictions):
    print(f"Original : {orig}")
    print(f"Corrected: {corrected}\n")