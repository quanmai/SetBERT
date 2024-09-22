from dataclasses import dataclass
from transformers import DataCollatorWithPadding
import torch
import jsonlines
import json

def read_jsonl(file_path):
    with jsonlines.open(file_path) as reader:
        return [obj for obj in reader]

class BooleanDataset(torch.utils.data.Dataset):
    def __init__(self,file_path):
        self.data = read_jsonl(file_path)
        print(f"load {len(self.data)} samples from {file_path}")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]


@dataclass
class BooleanCollator(DataCollatorWithPadding):
    max_len = 128

    def __call__(self, examples):
        G = [x['G'] for x in examples]
        P = [x['P'][0] for x in examples]
        N = [x['N'][0] for x in examples]
        a = self.tokenizer(
            G, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len,
            return_tensors='pt',
        )
        p = self.tokenizer(
            P, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len,
            return_tensors='pt',
        )
        n = self.tokenizer(
            N, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len,
            return_tensors='pt',
        )

        return {
            'a_input_ids': a['input_ids'],
            'a_attention_mask': a['attention_mask'],
            'p_input_ids': p['input_ids'],
            'p_attention_mask': p['attention_mask'],
            'n_input_ids': n['input_ids'],
            'n_attention_mask': n['attention_mask'],
        }
    
@dataclass
class MultiBooleanCollator(DataCollatorWithPadding):
    max_len = 128
    num_positives = 2
    num_negatives = 2

    def __call__(self, examples):
        G = [x['G'] for x in examples]
        P = [x['P'][0] for x in examples]
        MP = [x['P'][: self.num_positives] for x in examples]
        N = [x['N'][0] for x in examples]
        MN = [x['N'][: self.num_negatives] for x in examples]
        a = self.tokenizer(
            G, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len,
            return_tensors='pt',
        )
        p = self.tokenizer(
            P, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len,
            return_tensors='pt',
        )
        n = self.tokenizer(
            N, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len,
            return_tensors='pt',
        )
        mp = self.tokenizer(
            MP, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len,
            return_tensors='pt',
        )
        mn = self.tokenizer(
            MN, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len,
            return_tensors='pt',
        )

        return {
            'a_input_ids': a['input_ids'],
            'a_attention_mask': a['attention_mask'],
            'p_input_ids': p['input_ids'],
            'p_attention_mask': p['attention_mask'],
            'n_input_ids': n['input_ids'],
            'n_attention_mask': n['attention_mask'],
            'mp_input_ids': mp['input_ids'],
            'mp_attention_mask': mp['attention_mask'],
            'mn_input_ids': mn['input_ids'],
            'mn_attention_mask': mn['attention_mask'],
        }
