from transformers import HfArgumentParser, set_seed, EvalPrediction
from transformers import AutoConfig, AutoTokenizer
from arguments import ModelArguments, DataArguments, SBTrainingArguments
from model import SetBERT
from data_utils import BooleanDataset, BooleanCollator, MultiBooleanCollator
from trainer import BooleanTrainer
import logging
import os
import torch
from callbacks import SaveBestModelCallback

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, SBTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = os.environ['LOCAL_RANK']
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if local_rank in [-1, 0] else logging.WARN,
    )

    logger.warning(
        "Process rank: %s, device: %s, distributed training: %s, 16-bits training: %s",
        local_rank,
        device,
        bool(local_rank != -1),
        training_args.fp16,
    )
    
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)

    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = SetBERT.from_pretrained(model_args, training_args, config=config)
    train_dataset = BooleanDataset(data_args.train_file)
    eval_dataset = BooleanDataset(data_args.eval_file)
    data_collator = MultiBooleanCollator(tokenizer=tokenizer) #BooleanCollator(tokenizer=tokenizer) if training_args.training_loss != "bi_contrastive" else MultiBooleanCollator(tokenizer=tokenizer)

    trainer: BooleanTrainer = BooleanTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[SaveBestModelCallback(training_args.output_dir)],
    )

    if training_args.do_train:
        trainer.train()
        # if training_args.training_loss != "contrastive":
        #     trainer.save_model()
        #     if trainer.is_world_process_zero():
        #         tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        print("Evaluating")
        res = trainer.evaluate()
        print(res)

if __name__ == "__main__":
    main()
