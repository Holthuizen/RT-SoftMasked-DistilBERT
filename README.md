# RT-SoftMasked-DistilBERT
Implementation of [Zhang at el Spelling Error Correction with Soft-Masked BERT](https://aclanthology.org/2020.acl-main.82/)  CSC model, adopted for the task of RWEC and English (phonographic) languages. Its a proof of concept model with limited training and evaluation. 

### Preliminary findings
From the preliminary results its clear that this dual-model "detect and correct" approach by targeted noising and denoising of text, has potential for semi-flunt text rewriting using a pre-trained decoder. 

## Replicated model architecture proposed by Zhang at el. 
![Architecture](https://github.com/Holthuizen/RT-SoftMasked-DistilBERT/blob/main/SoftMaskBERT.png)
