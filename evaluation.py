#compare X, Y sequences. 
import torch
from tqdm import tqdm
import pandas as pd
from RWEC import soft_mask_encoding_decoding_pipeline
from correction_model import SoftMaskDistilBert
from detection_model import BiGRUDetectionNetwork

#SETUP: models, device and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detection_model = BiGRUDetectionNetwork().to(device)
detection_model.load_state_dict(torch.load("models/bigru_detection_model.pt"))

correction_model = SoftMaskDistilBert().to(device)
correction_model.eval()

tokenizer = correction_model.tokenizer

#LOAD TRAINING DATA: from CSV into pandas dataframe
dataset = "datasets/synthetic_typos_title_er0.15.csv"
df = pd.read_csv(dataset, comment="#")

#run the RWEC pipeline and store the results in the dataframes under "predictions"

T = df["X_perturbed"].to_list()
batch_size = 64
all_predictions = []

for i in tqdm(range(0, len(T), batch_size), desc="Predicting"):
    #inference
    text_predictions = soft_mask_encoding_decoding_pipeline (
        T[i:i + batch_size], 
        detection_model, 
        correction_model, 
        tokenizer
        )
    
    #append processed batch of size T[i:i + batch_size]
    all_predictions.extend(text_predictions)
    
df["predictions"] = all_predictions



# Force all columns to be strings and replace NaNs with empty strings
df["Y_target"] = df["Y_target"].fillna("").astype(str)
df["predictions"] = df["predictions"].fillna("").astype(str)
df["X_perturbed"] = df["X_perturbed"].fillna("").astype(str)


#METRICS: ASR style word based edit distance evaluation 
import jiwer

#text pre-processing to normalize text before comparing
transforms = jiwer.Compose([
    jiwer.ToLowerCase(), 
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords()
])

wer_score = jiwer.wer(
    df["Y_target"].tolist(), 
    df["predictions"].tolist(), 
    reference_transform=transforms, 
    hypothesis_transform=transforms
)

wer_baseline = jiwer.wer(
    df["Y_target"].tolist(), 
    df["X_perturbed"].tolist(), 
    reference_transform=transforms, 
    hypothesis_transform=transforms
)


print(f"Baseline WER: {wer_baseline} and model WER: {wer_score}")
pd.set_option('display.max_colwidth', None)
print(df.head(50))



output_file = "model_predictions.csv"

metadata_lines = []
with open(dataset, 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith("#"):
            metadata_lines.append(line)
        else:
            # As soon as we hit a line without a '#', we know the data started, so we stop reading.
            break


metadata_lines.append("# Model Architecture: Bi-GRU + Soft-Masked DistilBERT\n")
metadata_lines.append(f"# Baseline WER: {wer_baseline:.4f}\n")
metadata_lines.append(f"# Model WER: {wer_score:.4f}\n")

# 3. Write the assembled metadata to the new output file
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(metadata_lines)

df.to_csv(output_file, mode='a', index=False)
