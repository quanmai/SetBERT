import jsonlines
import faiss        
import numpy as np 
from tqdm import tqdm
from transformers import DPRQuestionEncoder,BertModel,BertTokenizer
import torch
import unicodedata
import time
import transformers
from example_utils import read_documents
transformers.logging.set_verbosity_error()
import example_utils


def normalize_query(question: str) -> str:
    question = question.replace("’", "'")
    return question

def normalize(text):
    return unicodedata.normalize("NFD", text)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents_file",default="../../../data/documents.jsonl")
    parser.add_argument("--test_file",default="../../../data/test.jsonl")
    parser.add_argument("--encoding_batch_size",type=int,default=32)
    parser.add_argument("--num_shards",type=int,default=2)
    parser.add_argument("--max_query_length",type=int,default=64)
    parser.add_argument("--num_docs",type=int,default=325505)
    parser.add_argument("--topk",type=int,default=1000)
    parser.add_argument("--embedding_dir",required=True)
    parser.add_argument("--pretrained_model_path",required=True)
    parser.add_argument("--safe_tensors",action='store_true',default=False)
    args = parser.parse_args()

    # doc title to idx
    documents = read_documents(args.documents_file)
    doc_2_idx = {doc.title: idx for idx, doc in enumerate(documents)}
    idx_2_doc = {idx: doc.title for idx, doc in enumerate(documents)}
    idx_2_len = {idx: len(doc.text.split()) for idx, doc in enumerate(documents)}

    ## load dataset
    queries,relevant_docs = [],[]
    num_queries = 0
    with jsonlines.open(args.test_file) as reader:
        for line in reader:
            num_queries += 1
            relevant_docs.append(
                {
                    "query": line['query'],
                    "docs": [doc_2_idx[doc] for doc in line['docs']],
                    "docs_len": [idx_2_len[doc_2_idx[doc]] for doc in line['docs']],
                    # "generated_query": line['gpt_generated_query']
                }
            )
            queries.append(normalize_query(line['query']))

    idx_2_query = {idx: query for idx, query in enumerate(queries)}
    
    queries = [
        queries[
            idx:idx+args.encoding_batch_size
        ] for idx in range(0,len(queries),args.encoding_batch_size)
    ]
    
    # make faiss index
    embedding_dimension = 1024
    index = faiss.IndexFlatIP(embedding_dimension)
    for idx in tqdm(range(args.num_shards),desc='building index from embedding...'):
        data = np.load(f"{args.embedding_dir}/documents_shard_{idx}.npy")
        index.add(data)
    
    ## load query encoder
    query_encoder = BertModel.from_pretrained(args.pretrained_model_path, add_pooling_layer=False, use_safetensors=args.safe_tensors)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_encoder.to(device).eval()

    # get query embeddings
    query_embeddings = []    
    for query in tqdm(queries,desc='encoding queries...'):
        with torch.no_grad():
            encoder_input = tokenizer(
                query,
                max_length=args.max_query_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
                ).to(device)
            query_embedding = query_encoder(**encoder_input)
        if isinstance(query_encoder,DPRQuestionEncoder):
            query_embedding = query_embedding.pooler_output
        else:
            query_embedding = query_embedding.last_hidden_state[:,0,:]
        query_embeddings.append(query_embedding.cpu().detach().numpy())
    query_embeddings = np.concatenate(query_embeddings,axis=0)

    ## retrieve top-k documents
    # print("searching index ", end=' ')
    start_time = time.time()
    D, I = index.search(query_embeddings, args.topk) # I: (num_queries, topk)
    predictions = []
    test_examples = example_utils.read_examples(args.test_file)
    for example, retrieved_docs in zip(test_examples, I):
        predictions.append(
            example_utils.Example(
                query=example.query,
                docs=[idx_2_doc[doc_idx] for doc_idx in retrieved_docs],
                original_query=example.original_query,
                metadata=example.metadata
            )
        )
    
    # Recall@K and MRecall@K
    # example_utils.write_examples("predictions.jsonl", predictions)
    K = [20, 50, 100, 1000]
    mrecall_vals = {k: [] for k in K}
    recall_vals = {k: [] for k in K}
    query_to_pred_examples = {example.query: example for example in predictions}
    p_vals, r_vals, f1_vals = [], [], []
    for gold_example in test_examples:
        pred_examples = query_to_pred_examples[gold_example.query]
        pred_docs = set(pred_examples.docs)
        gold_docs = set(gold_example.docs)
        # tp = len(gold_docs.intersection(pred_docs))
        # fp = len(pred_docs.difference(gold_docs))
        # fn = len(gold_docs.difference(pred_docs))
        # if tp:
        #     precision = tp / (tp + fp)
        #     recall = tp / (tp + fn)
        #     f1 = 2 * precision * recall / (precision + recall)
        # else:
        #     precision = 0.0
        #     recall = 0.0
        #     f1 = 0.0
        # p_vals.append(precision)
        # r_vals.append(recall)
        # f1_vals.append(f1)

        for k in K:
            pred_docs = set(pred_examples.docs[:k])
            if gold_docs.issubset(pred_docs):
                mrecall_vals[k].append(1.0)
            else:
                mrecall_vals[k].append(0.0)
            recall = len(gold_docs.intersection(pred_docs)) / len(gold_docs)
            recall_vals[k].append(recall)
    for k in K:
        print(f"Recall@{k}")
        example_utils.print_avg(test_examples, recall_vals[k])
        example_utils.print_avg_by_template(test_examples, recall_vals[k])
        print(f"MRecall@{k}")
        example_utils.print_avg(test_examples, mrecall_vals[k])
        example_utils.print_avg_by_template(test_examples, mrecall_vals[k])

    # print("Avg. Recall")
    # example_utils.print_avg(test_examples, r_vals)
    # example_utils.print_avg_by_template(test_examples, r_vals)
    # print("Avg. Precision")
    # example_utils.print_avg(test_examples, p_vals)
    # example_utils.print_avg_by_template(test_examples, p_vals)
    # print("Avg. F1")
    # example_utils.print_avg(test_examples, f1_vals)
    # example_utils.print_avg_by_template(test_examples, f1_vals)
    # p_vals,r_vals,f1_vals = _eval(relevant_docs, I, idx_2_query)
    # print(f"Evaluation results for k = {args.topk}")
    # print(f"Avg. Precision: {np.mean(p_vals)}")
    # print(f"Avg. Recall: {np.mean(r_vals)}")
    # print(f"Avg. F1: {np.mean(f1_vals)}")