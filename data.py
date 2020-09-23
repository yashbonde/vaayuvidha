# all shit realted to data --> parsing --> cleaning --> generation --> storage
# takes its own arguments
# 15.09.2020 - @yashbonde

import re
import json
import numpy as np
import pandas as pd
import sentencepiece as spm

import torch
from torch.utils.data import Dataset

from vocab import Mappings, Fields

FC = Fields()
MC = Mappings()

# ---- functions ---- #
def create_sp_tokenizer(fpath, model, vocab_size, normalization_rule_name="nfkc_cf", split_by_whitespace=False):
    split_by_whitespace = "true" if split_by_whitespace else "false"
    spm.SentencePieceTrainer.train(f'''--input={fpath} --model_prefix={model} --vocab_size={vocab_size}\
        --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3\
        --user_defined_symbols={",".join(FC.sp_tokens)}\
        --pad_piece={FC.PAD}\
        --unk_piece={FC.UNK}\
        --bos_piece={FC.BOS}\
        --eos_piece={FC.EOS}\
        --normalization_rule_name={normalization_rule_name}\
        --split_by_whitespace={split_by_whitespace}''')
    
    
def load_tokenizer(model):
    return spm.SentencePieceProcessor(f"{model}.model")


# ---- main class ---- #
class WeatherModelingDataset(Dataset):
    def __init__(self, config, mode = "train"):
        self.config = config

        print("Loading samples ...")
        with open(config.path, "r") as f:
            data = json.load(f)

        self.data = [data[i:i+config.maxlen] for i in range(len(data) - config.maxlen)]
        print(f"Dataset [{mode}] {len(self.data)} samples")

    def __len__(self):
        return len(self.data)
    
  
    def __getitem__(self, idx, **kwargs):
        # print("\n\n\n\n")
        sample = self.data[idx]

        # parse to graph then

        # huggingface transformers should be able to handle the attention mask by itself
        return {"input_ids": torch.from_numpy(np.asarray(ids))}


class DatasetConfig:
    path = None
    sheets = None
    size = 3000 # since we generate data runtime size doesn't matter

    # about probabilities of the fields
    pf = 0.5 # probability of adding fields in sample
    fmax = 0.8 # at max include only 80 of fields
    fmin = 0.1 # atleast include 10 of fields
    
    proc = "sample"

    maxlen = None # what is the maximum length of sequence to return

    def __init__(self, **kwargs):
        self.attrs = []
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

    def __repr__(self):
        return "---- DATASET CONFIGURATION ----\n" + \
            "\n".join([f"{k}\t{getattr(self, k)}" for k in list(set([
                "path",
                "sheets",
                "size",
                "pf",
                "fmax",
                "fmin",
                "maxlen",
                "proc"
            ] + self.attrs)) 
        ]) + "\n"
