import argparse
import logging
import random
from utils.example_utils import (
    read_examples,
    read_documents,
    read_documents_flatten,
    Document,
    TrainingExample,
)
from utils import generate_negative_query
from gpt.oai import gpt_generate_query # query expansion using gpt
from tqdm import tqdm

logger = logging.getLogger()

def gen_training_examples(
    fdocument: str,
    fexamples: str,
    fbm25: str,
    num_hard_negatives: int,
    num_random_negatives: int,
    output: str,
    gpt_generate: bool = False,
) -> None:
    documents = read_documents(fdocument)
    examples = read_examples(fexamples)
    bm25output = read_examples(fbm25)

    # Map document title to text
    doc_title_to_text = {doc.title: doc.text for doc in documents}
    outputs = []
    encoding_progress_bar = tqdm(
        total=len(examples),
        ncols=100,
        desc="Generating training examples...",
    )
    for example, predictions in zip(examples, bm25output):
        logger.info("Processing example %s." % example.query)
        relevant_titles = set(example.docs)
        if len(relevant_titles) == 0:
            raise ValueError(f"Missing relevant documents for query: `{example.query}`")
        
        # Add relevant documents as positive examples.
        pos_docs = [
            Document(
                title=doc_title,
                text=doc_title_to_text[doc_title],
            ) for doc_title in relevant_titles
        ]

        # Add BM25 negative examples as hard negatives.
        hard_neg_docs = []
        for doc_title in predictions.docs:
            if doc_title not in relevant_titles:
                if doc_title not in doc_title_to_text:
                    raise ValueError(f"Missing document title: `{doc_title}`")
                hard_neg_docs.append(
                    Document(
                        title=doc_title,
                        text=doc_title_to_text[doc_title],
                    )
                )
            if len(hard_neg_docs) >= num_hard_negatives:
                break    
    
        # Add random non-relevant examples as negative documents.
        neg_docs = []
        random_titles = random.sample(
            list(doc_title_to_text.keys()),
            k = 2 * num_random_negatives,
        )
        for doc_title in random_titles:
            if doc_title not in relevant_titles:
                neg_docs.append(
                    Document(
                        title=doc_title,
                        text=doc_title_to_text[doc_title],
                    )
                )
            if len(neg_docs) >= num_random_negatives:
                break
        outputs.append(
            TrainingExample(
                query=example.query,
                positive_docs=pos_docs,
                negative_docs=neg_docs,
                hard_negative_docs=hard_neg_docs,
            )
        )
        encoding_progress_bar.update(1)

    import jsonpickle
    output_pickpled = jsonpickle.encode(outputs, unpicklable=False)
    with open(output, "w") as json_file:
        json_file.write(output_pickpled)

    print(f"Data has been dump to {output}")

def gen_training_examples_flatten(
    fdocument: str, 
    fexamples: str,
    fbm25: str,
    num_hard_negatives: int,
    num_random_negatives: int,
    min_num_positives: int,
    output: str,
    num_neg_queries: int = 0,
    num_hard_neg_queries: int = 0,
    gpt_generate: bool = False,
    gen_test: bool = False,
    ):

    documents = read_documents_flatten(fdocument)
    examples = read_examples(fexamples)
    bm25output = read_examples(fbm25)

    # Map document title to text
    doc_title_to_text = {doc["title"]: doc["text"] for doc in documents}
    encoding_progress_bar = tqdm(
        total=len(examples),
        ncols=100,
        desc="Generating training examples...",
    )
    outputs = []
    for example, predictions in zip(examples, bm25output):
        logger.info("Processing example %s." % example.query)
        relevant_titles = set(example.docs)
        if len(relevant_titles) == 0:
            # raise ValueError(f"Missing relevant documents for query: `{example.query}`")
            continue

        # Add relevant documents as positive examples.
        pos_docs = [
            {
                "title": doc_title,
                "text": doc_title_to_text[doc_title],
            } for doc_title in relevant_titles
        ]

        # 
        if gpt_generate:
            gpt_generated_query = gpt_generate_query(example.query)
        if not gen_test:  # only for training and dev
            # Pad positive examples if necessary.
            if len(pos_docs) < min_num_positives:
                pos_docs += [
                    pos_docs[0] for _ in range(min_num_positives - len(pos_docs))
                ]

            # Add BM25 negative examples as hard negatives.
            hard_neg_docs = []
            for doc_title in predictions.docs:
                if doc_title not in relevant_titles:
                    if doc_title not in doc_title_to_text:
                        raise ValueError(f"Missing document title: `{doc_title}`")
                    hard_neg_docs.append(
                        {
                            "title": doc_title,
                            "text": doc_title_to_text[doc_title],
                        }
                    )
                if len(hard_neg_docs) == num_hard_negatives:
                    break    
            assert len(hard_neg_docs) == num_hard_negatives
            # Add random non-relevant examples as negative documents.
            neg_docs = []
            random_titles = random.sample(
                list(doc_title_to_text.keys()),
                k = 2 * num_random_negatives,
            )
            for doc_title in random_titles:
                if doc_title not in relevant_titles:
                    neg_docs.append(
                        {
                            "title": doc_title,
                            "text": doc_title_to_text[doc_title],
                        }
                    )
                if len(neg_docs) == num_random_negatives:
                    break
            assert len(neg_docs) == num_random_negatives

            # Add negative queries
            if num_neg_queries > 0 and not gen_test:
                neg_queries = random.sample(
                    [e.query for e in examples if e.query != example.query],
                    num_neg_queries,
                )

                hard_neg_queries = generate_negative_query(
                    example.original_query,
                    example.metadata.template,
                )

                # if example.metadata.template == "_":
                if len(hard_neg_queries) < num_hard_neg_queries:
                    hard_neg_queries_sampled = random.sample(
                        [e.query for e in examples if e.query not in set([example.query] + neg_queries)],
                        num_hard_neg_queries - len(hard_neg_queries),
                    )
                    hard_neg_queries = hard_neg_queries + hard_neg_queries_sampled
            
            outputs.append(
                {
                    "query": example.query,
                    "original_query": example.original_query,
                    "template": example.metadata.template,
                    "positive_docs": pos_docs,
                    "negative_docs": neg_docs,
                    "hard_negative_docs": hard_neg_docs,
                    "negative_queries": neg_queries if num_neg_queries > 0 else [],
                    "hard_negative_queries": hard_neg_queries if num_hard_neg_queries > 0 else [],
                    "gpt_generated_query": gpt_generated_query if gpt_generate else "",
                }
            )
        else:
            outputs.append(
                {
                    "query": example.query,
                    "positive_docs": pos_docs,
                    "original_query": example.original_query,
                    "template": example.metadata.template,
                    "gpt_generated_query": gpt_generated_query if gpt_generate else "",
                }
            )
        print(example.query)
        if gpt_generate:
            print(f"-> {gpt_generated_query}")
        encoding_progress_bar.update(1)

    import json
    with open(output, "w") as json_file:
        json_file.write(json.dumps(outputs, indent=4) + "\n")

    print(f"Data has been dump to {output}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fdocument",
        default="../quest/dataset/documents.jsonl",
        type=str,
    )
    parser.add_argument(
        "--fbm25",
        default="../quest/output/bm25output.jsonl",
        type=str,
    )
    parser.add_argument(
        "--num-hard-negatives",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--num-random-negatives",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--gen-dev",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--gen-test",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    output = "./data/training_negative_queries_gpt_generated.json"
    fexamples = "../quest/dataset/train.jsonl"
    if args.gen_dev:
        output = "./data/dev_negative_queries_gpt_generated.json"
        fexamples = "../quest/dataset/val.jsonl"
    elif args.gen_test:
        output = "./data/test_example_gpt_generated.json"
        fexamples = "../quest/dataset/test.jsonl"
        gen_training_examples_flatten(
            fdocument=args.fdocument,
            fexamples=fexamples,
            fbm25=args.fbm25,
            num_hard_negatives=args.num_hard_negatives,
            num_random_negatives=args.num_random_negatives,
            min_num_positives=10,
            output=output,
            num_neg_queries=10,
            num_hard_neg_queries=10,
            gpt_generate=True,
            gen_test=True,
        )
        return 

    if args.flatten:
        gen_training_examples_flatten(
            fdocument=args.fdocument,
            fexamples=fexamples,
            fbm25=args.fbm25,
            num_hard_negatives=args.num_hard_negatives,
            num_random_negatives=args.num_random_negatives,
            min_num_positives=10,
            output=output,
            num_neg_queries=10,
            num_hard_neg_queries=10,
            gpt_generate=True,
        )
    else:
        gen_training_examples(
            fdocument=args.fdocument,
            fexamples=fexamples,
            fbm25=args.fbm25,
            num_hard_negatives=args.num_hard_negatives,
            num_random_negatives=args.num_random_negatives,
            output=output,
        )


if __name__ == "__main__":
    main()
