import os
import jsonlines
import numpy as np
import torch
from tqdm import tqdm
from accelerate import PartialState
from transformers import (
    BertTokenizer,
    BertModel,
)
import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoConfig


def normalize_document(document: str):
    document = document.replace("\n", " ").replace("’", "'").replace("'''", "")
    if document.startswith('"'):
        document = document[1:]
    if document.endswith('"'):
        document = document[:-1]
    return document

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--document_path", default="../data/documents.jsonl")
    parser.add_argument("--num_docs", type=int, default=325505)
    parser.add_argument("--encoding_batch_size", type=int, default=1024)
    parser.add_argument("--pretrained_model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sequence_len", type=int, default=256)
    parser.add_argument("--safetensor", action='store_true', default=False)
    parser.add_argument("--hf_bert", default="bert-base-uncased")
    args = parser.parse_args()

    distributed_state = PartialState()
    device = distributed_state.device

    # Load encoder
    if args.safetensor:
        # config = AutoConfig.from_pretrained(args.hf_bert)
        doc_encoder = BertModel.from_pretrained(args.pretrained_model_path, add_pooling_layer=False, use_safetensors=True)
    else:
        doc_encoder = BertModel.from_pretrained(args.pretrained_model_path, add_pooling_layer=False)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    doc_encoder.eval()
    doc_encoder.to(device)

    # Load document passages
    progress_bar = tqdm(
        total=args.num_docs, 
        disable=not distributed_state.is_main_process, 
        ncols=100, 
        desc='loading document passages...'
    )

    documents = []
    with jsonlines.open(args.document_path) as reader:
        for line in reader:
            documents.append([line['title'], normalize_document(line['text'])])
            progress_bar.update(1)
    

    with distributed_state.split_between_processes(documents) as sharded_documents:
        sharded_documents = [
            sharded_documents[idx:idx+args.encoding_batch_size]
            for idx in range(0, len(sharded_documents), args.encoding_batch_size)
        ]
        encoding_progress_bar = tqdm(
            total=len(sharded_documents), 
            disable=not distributed_state.is_main_process, 
            ncols=100, 
            desc='encoding test data...'
        )
        doc_embeddings = []
        for doc in sharded_documents:
            title = [x[0] for x in doc]
            passage = [x[1] for x in doc]
            model_input = tokenizer(
                title, 
                passage, 
                max_length=args.sequence_len, 
                padding='max_length', 
                return_tensors='pt', 
                truncation=True
            ).to(device)
            with torch.no_grad():
                if isinstance(doc_encoder, BertModel):
                    CLS_POS = 0
                    output = doc_encoder(**model_input).last_hidden_state[:, CLS_POS, :]
                    output = output.cpu().numpy()
                else:
                    output = doc_encoder(**model_input).pooler_output.cpu().numpy()
            doc_embeddings.append(output)
            encoding_progress_bar.update(1)
        doc_embeddings = np.concatenate(doc_embeddings, axis=0)
        # normalize
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(f'{args.output_dir}/documents_shard_{distributed_state.process_index}.npy', doc_embeddings)
