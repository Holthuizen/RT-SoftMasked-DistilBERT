import torch
from transformers import DistilBertTokenizer
from symspellpy import SymSpell, Verbosity
from tqdm import tqdm

def build_token_edit_distance_map(save_path="models/token_edit_map.pt"):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    vocab = tokenizer.get_vocab() # Dict of {string: id}
    id_to_token = {v: k for k, v in vocab.items()}
    
    # Initialize SymSpell
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    
    print("Loading vocabulary into SymSpell...")
    for token_str in vocab.keys():
        # DistilBERT subwords start with '##'. We strip it for distance calcs
        clean_str = token_str.replace("##", "")
        # Add to symspell with a dummy frequency of 1
        if len(clean_str) > 1: # Ignore single characters/punctuation
            sym_spell.create_dictionary_entry(clean_str, 1)
            
    print("Building ID mapping...")
    token_map = {}
    
    # Start at 1000 to skip special tokens [PAD], [CLS], and basic punctuation
    for token_id in tqdm(range(1000, tokenizer.vocab_size)):
        token_str = id_to_token[token_id]
        clean_str = token_str.replace("##", "")
        
        # Skip very short tokens to avoid explosive matching
        if len(clean_str) < 3:
            continue
            
        suggestions = sym_spell.lookup(clean_str, Verbosity.ALL, max_edit_distance=2)
        
        # Convert matched strings back to Token IDs
        matched_ids = []
        for sym in suggestions:
            match_str = sym.term
            
            # Re-attach '##' if the original token was a subword
            if token_str.startswith("##"):
                lookup_str = "##" + match_str
            else:
                lookup_str = match_str
                
            if lookup_str in vocab and vocab[lookup_str] != token_id:
                matched_ids.append(vocab[lookup_str])
                
        if matched_ids:
            token_map[token_id] = matched_ids

    torch.save(token_map, save_path)
    print(f"Map saved! {len(token_map)} tokens have valid edit-distance replacements.")

if __name__ == "__main__":
    build_token_edit_distance_map()