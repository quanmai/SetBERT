from transformers import HfArgumentParser, set_seed
from transformers import AutoConfig, AutoTokenizer, BertModel
from arguments import ModelArguments, DataArguments, DPRTrainingArguments
from model import SetDPR
from data_utils import DPRDataset, DPRCollator
from trainer import DPRTrainer
from callbacks import SaveBiEncoderCallback, SaveModelAndTokenizerCallback
import logging
import os
import torch

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, DPRTrainingArguments))
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

    config = AutoConfig.from_pretrained(model_args.hf_bert) if "bert" in model_args.query_encoder_model_name_or_path else None
    tokenizer = AutoTokenizer.from_pretrained(model_args.hf_bert
                                              if "bert" in model_args.query_encoder_model_name_or_path 
                                              else model_args.query_encoder_model_name_or_path )
    model = SetDPR.from_pretrained(
        model_args, 
        config=config,
    )
    train_dataset = DPRDataset(data_args.train_file)
    eval_dataset = DPRDataset(data_args.eval_file)
    data_collator = DPRCollator(tokenizer=tokenizer)

    trainer: DPRTrainer = DPRTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[SaveBiEncoderCallback(training_args.output_dir), SaveModelAndTokenizerCallback(training_args.output_dir)],
    )

    if training_args.do_train:
        trainer.train()

    if training_args.do_eval:
        print("Evaluating")
        res = trainer.evaluate()
        print(res)

if __name__ == "__main__":
    main()
