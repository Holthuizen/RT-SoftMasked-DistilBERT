# RT-SoftMasked-DistilBERT
Implementation of [Zhang at el Spelling Error Correction with Soft-Masked BERT](https://aclanthology.org/2020.acl-main.82/) paper, adopted for the task of RWEC and English (phonographic) languages. Its a proof of concept model with limited training and evaluation. 

### preliminary findings
From the preliminary results its clear that this dual-model detect and correct approach by targeted noising and denoising has potential for more fluent text rewriting using a pre-trained decoder. 
The question is now to see what the effect of further finetuning of the detector and correction model could achieve.


## Replicated model architecture proposed by Zhang at el. 
![Architecture](https://github.com/Holthuizen/RT-SoftMasked-DistilBERT/blob/main/RWEC_generation.py)
