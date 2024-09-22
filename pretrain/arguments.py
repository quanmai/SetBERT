from dataclasses import dataclass, field
from transformers import TrainingArguments
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    emb_dim: int = field(
        default=768,
        metadata={"help": "Dimension of the embeddings"}
    )
    project: bool = field(
        default=False,
        metadata={"help": "Project embeddings to emb_dim"}
    )


@dataclass
class DataArguments:
    train_file: str = field(
        default="../dataset/train.jsonl",
        metadata={"help": "Path to training data"}
    )
    eval_file: str = field(
        default="../dataset/eval.jsonl",
        metadata={"help": "Path to evaluation data"}
    )
    
@dataclass
class SBTrainingArguments(TrainingArguments):
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of steps for linear warmup"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay for optimizer"}
    )
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Learning rate for optimizer"}
    )
    num_train_epochs: int = field(
        default=5,
        metadata={"help": "Number of training epochs"}
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "Strategy for evaluation"}
    )
    eval_steps: int = field(
        default=1000,
        metadata={"help": "Number of steps between evaluations"}
    )
    logging_steps: int = field(
        default=1000,
        metadata={"help": "Number of steps between logging"}
    )
    save_steps: int = field(
        default=1000,
        metadata={"help": "Number of steps between saves"}
    )
    per_device_train_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size per device"}
    )
    per_device_eval_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size per device for evaluation"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of steps before optimizer step"}
    )
    logging_dir: str = field(
        default="./logs",
        metadata={"help": "Directory for logs"}
    )
    output_dir: str = field(
        default="./output",
        metadata={"help": "Directory for output"}
    )
    training_loss: str = field(
        default="triplet",
        metadata={"help": "Loss function for training"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Use 16-bit precision"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    save_only_model : bool = field(
        default=True,
        metadata={"help": "Save only model"}
    )
    save_strategy: str = field(
        default="no",
        metadata={"help": "Strategy for saving"}
    )