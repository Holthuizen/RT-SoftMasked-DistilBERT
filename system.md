





1) sentence-transformers/agnews ---> RWEC_GEN_V2 ---> (X,Y,E) pairs csv (synthetic_typos_title_er0.15)

2) detection_model_training: load and defines BI-GRU model which is fine-tuned on synthetic_typos_title_er. 

3) correction_model: defines custom BERT MODEl



todo

[] switch data to descriptions
[] expand model definition
[] add residual 
[] debugging and logging
[] evaluation incorporate scribendi_score and gleu
[] evaluate base model 
[] use residual model as correction model 
