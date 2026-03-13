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
df = pd.read_csv("datasets/synthetic_typos_title_er0.15.csv")

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
print(df.head(20))