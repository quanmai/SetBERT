from dataclasses import dataclass
import jsonlines
import json
import typing
from typing import List, Optional, Dict


@dataclass
class ExampleMetadata:
    """Optional metadata used for analysis."""
    # The template used to synthesize this example.
    template: Optional[str] = None
    # The domain of the example (e.g. films, books, plants, animals).
    domain: Optional[str] = None
    # Fluency labels
    fluency: Optional[typing.Sequence[bool]] = None
    # Meaning labels
    meaning: Optional[typing.Sequence[bool]] = None
    # Naturalness labels
    naturalness: Optional[typing.Sequence[bool]] = None
    # The following fields are dictionaries keyed by document title.
    # The sequences can contain multiple values for replicated annotations.
    relevance_ratings: Optional[typing.Dict[str, typing.Sequence[str]]] = None
    evidence_ratings: Optional[typing.Dict[str, typing.Sequence[str]]] = None
    # The nested value is a map from query substring to document substring.
    attributions: Optional[typing.Dict[str, typing.Sequence[typing.Dict[str, str]]]] = None

@dataclass
class Example:
    """Represents a query paired with a set of documents."""
    query: str
    docs: typing.Iterable[str]
    original_query: Optional[str] = None
    # Scores can be optionally included if the examples are generated from model
    # predictions. Indexes of `scores` should correspond to indexes of `docs`.
    scores: Optional[typing.Iterable[float]] = None
    # Optional metadata.
    metadata: Optional[ExampleMetadata] = None

# Extract examples from jsonl file
def read_examples(filepath: str) -> List[Example]:
    examples_json = read_jsonl(filepath)
    examples = [Example(**example) for example in examples_json]
    for example in examples:
        example.metadata = ExampleMetadata(**example.metadata)
    return examples

def read_jsonl(file: str):
    """Read jsonl file to a List of Dicts."""
    data = []
    with jsonlines.open(file, mode="r") as jsonl_reader:
        for json_line in jsonl_reader:
            try:
                data.append(json_line)
            except jsonlines.InvalidLineError as e:
                print("Failed to parse line: `%s`" % json_line)
                raise e
    print("Loaded %s lines from %s." % (len(data), file))
    return data

def write_examples(filepath: str, examples: List[Example]):
    examples_json = [example.__dict__ for example in examples]
    for example in examples_json:
        example["metadata"] = example["metadata"].__dict__
    with open(filepath, "w") as file:
        for example in examples_json:
            file.write(json.dumps(example) + "\n")
        

# Read document
@dataclass
class Document:
    """Represents a document with its title and text."""
    # Document title (should be unique in corpus).
    title: str
    # Document text.
    text: str


def read_documents(filepath: str) -> List[Document]:
    documents_json = read_jsonl(filepath)
    return [Document(**document) for document in documents_json]

def read_documents_flatten(filepath: str) -> List[Dict]:
    documents_json = read_jsonl(filepath)
    return [
        dict(
            title=document["title"],
            text=document["text"],
        ) for document in documents_json
    ]


# Write file
def write_jsonl(filepath, data):
    with jsonlines.open(filepath, mode="w") as jsonl_file:
        jsonl_file.write_all(data)
    print(f"Data has been dump to {filepath}")

def write_json(filepath, data):
    with open(filepath, "w") as json_file:
        json.dump(data, json_file)
    print(f"Data has been dump to {filepath}")

@dataclass
class TrainingExample:
    """Represents a training example for a retrieval model."""
    query: str
    positive_docs: List[Document]
    negative_docs: List[Document]
    hard_negative_docs: List[Document]

# def normalize(text: str):
#     text = text.replace("\n", " ")

import collections


def print_avg_by_key(examples, metrics, key_fn):
    """Prints avg results partitioned according to some `key_fn`.

    Args:
        examples: List of Example instances.
        metrics: List of float values corresponding to `examples`.
        key_fn: Function that takes an Example and returns a key to aggregate over.
    """
    key_to_metrics = collections.defaultdict(list)
    for example, metric in zip(examples, metrics):
        key = key_fn(example)
        key_to_metrics[key].append(metric)
    # Compute averages.
    key_to_avg = {
        key: sum(vals) / len(vals) for key, vals in key_to_metrics.items()
    }
    for key, val in key_to_avg.items():
        print("%s (%s): %s" % (key, len(key_to_metrics[key]), val))


def print_avg_by_template(examples, metrics):
    """Prints results partitioned by template."""
    key_fn = lambda example: example.metadata.template
    return print_avg_by_key(examples, metrics, key_fn)


def print_avg(examples, metrics):
    key_fn = lambda unused_x: "all"
    return print_avg_by_key(examples, metrics, key_fn)
    