from dataclasses import dataclass
from transformers import DataCollatorWithPadding
import torch
import jsonlines
import json

def read_jsonl(file_path):
    with jsonlines.open(file_path) as reader:
        return [obj for obj in reader]

class DPRDataset(torch.utils.data.Dataset):
    def __init__(self,file_path):
        if file_path.endswith('.jsonl'):
            self.data = read_jsonl(file_path)
        elif file_path.endswith('.json'):
            self.data = json.load(open(file_path))
        else:
            raise ValueError(f"Unknown file format for {file_path}")
        print(f"load {len(self.data)} samples from {file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]


@dataclass
class DPRCollator(DataCollatorWithPadding):
    max_query_len = 64
    max_passage_len = 256
    num_hard_neg_ctxs = 5
    num_pos_ctxs = 1
    mode = 't5'

    def __call__(self, examples):
        positive_documents = [x['positive_docs'][: self.num_pos_ctxs] for x in examples]
        positive_titles = [x['title'] for xx in positive_documents for x in xx]
        positive_passages = [x['text'] for xx in positive_documents for x in xx]

        hard_negative_documents = [x['hard_negative_docs'][: self.num_hard_neg_ctxs] for x in examples]
        hard_negative_titles = [x['title'] for xx in hard_negative_documents for x in xx]
        hard_negative_passages = [x['text'] for xx in hard_negative_documents for x in xx]

        query = [x['query'] for x in examples]

        if self.mode == 't5':
            positive_batch = [f"{title}. {passage}" for title, passage in zip(positive_titles, positive_passages)]
            negative_batch = [f"{title}. {passage}" for title, passage in zip(hard_negative_titles, hard_negative_passages)]
            negative_inputs = self.tokenizer(
                negative_batch,
                truncation=True,
                padding='max_length',
                max_length=self.max_passage_len,
                return_tensors='pt',
            )
            positive_inputs = self.tokenizer(
                positive_batch,
                truncation=True,
                padding='max_length',
                max_length=self.max_passage_len,
                return_tensors='pt',
            )
        else:
            negative_inputs = self.tokenizer(
                hard_negative_titles,
                hard_negative_passages,
                truncation=True,
                padding='max_length',
                max_length=self.max_passage_len,
                return_tensors='pt',
            )
            positive_inputs = self.tokenizer(
                positive_titles,
                positive_passages,
                truncation=True,
                padding='max_length',
                max_length=self.max_passage_len,
                return_tensors='pt',
            )
        query_inputs = self.tokenizer(
            query,
            truncation=True,
            padding='max_length',
            max_length=self.max_query_len,
            return_tensors='pt',
        )

        return {
            'query_input': query_inputs,
            'positive_input': positive_inputs,
            'negative_input': negative_inputs,
        }

        
