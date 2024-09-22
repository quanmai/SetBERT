from dataclasses import dataclass, field
from transformers import TrainingArguments
@dataclass
class ModelArguments:
    hf_bert: str = field(
        default="bert-base-uncased",
        metadata={"help": "Huggingface model name"}
    )
    query_encoder_model_name_or_path: str = field(
        # default="bert-base-uncased",
        default="/home/quanmai/workspace/IR/mulRetrievers/gpt/setBert/pretrain/output/checkpoint_2024-06-01-1345.33/best_model/",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    document_encoder_model_name_or_path: str = field(
        default="/home/quanmai/workspace/IR/mulRetrievers/gpt/setBert/pretrain/output/checkpoint_2024-06-01-1345.33/best_model/",
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
        default="/home/quanmai/workspace/IR/mulRetrievers/data/training_negative_queries.json",
        metadata={"help": "Path to training data"}
    )
    eval_file: str = field(
        default="/home/quanmai/workspace/IR/mulRetrievers/data/dev_negative_queries.json",
        metadata={"help": "Path to evaluation data"}
    )
    
@dataclass
class DPRTrainingArguments(TrainingArguments):
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of steps for linear warmup"}
    )
    warmup_ratio: float = field(
        default=0,
        metadata={"help": "Ratio of steps for linear warmup"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay for optimizer"}
    )
    learning_rate: float = field(
        default=2e-5,
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
    fp16: bool = field(
        default=False,
        metadata={"help": "Use 16-bit precision"}
    )
    seed: int = field(
        default=42, #seeds = [42, 123, 456, 789, 101112, 0, 1213]
        metadata={"help": "Random seed"}
    )
    save_only_model : bool = field(
        default=True,
        metadata={"help": "Save only model"}
    )
    max_query_len: int = field(
        default=64,
        metadata={"help": "Maximum query length"}
    )
    max_passage_len: int = field(
        default=256,
        metadata={"help": "Maximum passage length"}
    )
    num_hard_neg_ctxs: int = field(
        default=5,
        metadata={"help": "Number of hard negative contexts"}
    )
    num_pos_ctxs: int = field(
        default=1,
        metadata={"help": "Number of positive contexts"}
    )
    save_strategy: str = field(
        default="no",
        metadata={"help": "Strategy for saving"}
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Load best model at the end"}
    )
    save_safetensors: bool = field(
        default=False,
        metadata={"help": "Save safetensors"}
    )
    adam_eps: float = field(
        default=1e-8,
        metadata={"help": "Adam epsilon"}
    )
    fp32: bool = field(
        default=False,
        metadata={"help": "Use 32-bit precision"}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Use 16-bit precision"}
    )