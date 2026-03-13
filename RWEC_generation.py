#disclaimer: produced by GEN AI

import re
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from symspellpy import SymSpell, Verbosity
import pkg_resources
import html



hyp_parms = {"text_field": "title", "error_rate": 0.15,"homophone_ratio":0.8, "max_rows": 125000}


# ==========================================
# MULTIPROCESSING GLOBALS
# ==========================================
worker_sym_spell = None
worker_vocab = None



def init_worker():
    global worker_sym_spell, worker_vocab
    worker_sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt"
    )
    worker_sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    worker_vocab = list(worker_sym_spell.words.keys())

# ==========================================
# PERTURBATION LOGIC
# ==========================================
def perturb_text(args: Tuple[str, float, float]) -> Tuple[str, str, List[int]]:
    """
    Returns a tuple of (X_perturbed, Y_target, error_mask).
    """
    text, error_rate, homophone_ratio = args
    
    # 1. Lowercase the entire string immediately
    text = text.lower()
    
    # Tokenize: Split by word boundaries but keep non-word separators 
    tokens = re.split(r'(\W+)', text)
    
    # Initialize a mask of all 0s, perfectly aligned with the tokens list
    mask = [0] * len(tokens)
    
    # Find indices of valid words (alphabetic and length > 1)
    word_indices = [i for i, token in enumerate(tokens) if token.isalpha() and len(token) > 1]
    
    num_errors = int(len(word_indices) * error_rate)
    
    # Return unchanged if the sentence is too short
    if num_errors == 0:
        return text, text, mask  
        
    indices_to_replace = random.sample(word_indices, num_errors)
    perturbed_tokens = tokens.copy()
    
    for idx in indices_to_replace:
        # Mark this specific token index as an error (1)
        mask[idx] = 1 
        
        original_word = tokens[idx]
        replacement = None
        
        # 80% chance to find a near-homophone
        if random.random() < homophone_ratio:
            suggestions = worker_sym_spell.lookup(
                original_word, # Already lowercased
                Verbosity.ALL, 
                max_edit_distance=2
            )
            candidates = [s.term for s in suggestions if s.term != original_word]
            
            if candidates:
                replacement = random.choice(candidates)
                
        # Fallback to random word
        if replacement is None:
            replacement = random.choice(worker_vocab)
            
        perturbed_tokens[idx] = replacement
        
    return "".join(perturbed_tokens), text, mask

# ==========================================
# MAIN GENERATOR PIPELINE
# ==========================================
def generate_dataset(
    dataset_name: str = "sentence-transformers/agnews",
    text_field: str = hyp_parms["text_field"],
    error_rate: float = hyp_parms["error_rate"],
    homophone_ratio: float = hyp_parms["homophone_ratio"],
    max_rows: int = min(hyp_parms["max_rows"], 120000), 
    num_workers: int = max(1, mp.cpu_count() - 1)
) -> pd.DataFrame:
    
    print(f"Loading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
        
    texts = dataset[text_field][:max_rows]
    
    # Clean HTML entities (e.g., &quot; becomes ")
    texts = [html.unescape(t) for t in texts if isinstance(t, str) and t.strip()]
    
    print(f"Extracted {len(texts)} {text_field}s. Generating errors using {num_workers} cores...")
    
    args = [(text, error_rate, homophone_ratio) for text in texts]
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        for res in tqdm(executor.map(perturb_text, args), total=len(args), desc="Perturbing Text"):
            results.append(res)
            
    # Add the mask to our DataFrame columns
    df = pd.DataFrame(results, columns=["X_perturbed", "Y_target", "error_mask"])
    
    # Clean up: Drop rows where the sentence was too short for any perturbation
    df = df[df['X_perturbed'] != df['Y_target']].reset_index(drop=True)
    
    print(f"Generation complete. Yielded {len(df)} corrupted pairs.")
    return df

if __name__ == "__main__":
    print(f"Starting process with {hyp_parms}")

    synthetic_data = generate_dataset(**hyp_parms)
    
    # Dynamic filename based on parameters
    filename = f"synthetic_typos_{hyp_parms['text_field']}_er{hyp_parms['error_rate']}.csv"
    
    # Save the output to a CSV
    synthetic_data.to_csv(filename, index=False)
    
    pd.set_option('display.max_colwidth', None)
    print(f"\nSaved to {filename}. Preview of Generated Data:")
    print(synthetic_data[10:15])