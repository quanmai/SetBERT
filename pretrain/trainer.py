from transformers.trainer import Trainer, nested_detach, EvalLoopOutput
from torch import nn
from typing import Dict, Union, Any, Tuple, Optional, List
import torch
from torch.utils.data import DataLoader
# from transformers import AdamW
import numpy as np
from torch.optim import AdamW

class BooleanTrainer(Trainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataloader.")
        train_sampler = self._get_train_sampler()

        train_dataloader =  DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )
        return self.accelerator.prepare(train_dataloader)

    def get_eval_dataloader(self, eval_dataset: Optional[torch.utils.data.Dataset] = None):
        if self.eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=None,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )
        return self.accelerator.prepare(eval_dataloader)
        
    def create_optimizer(self):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters, 
            lr=self.args.learning_rate, 
            betas=(0.9, 0.999), 
            eps=1e-8
        )
        return self.optimizer
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.args.warmup_ratio > 0:
            self.args.warmup_steps = num_training_steps * self.args.warmup_ratio

        super().create_optimizer_and_scheduler(num_training_steps)
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Tuple[Dict[str, Union[torch.Tensor, Any]]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            if self.args.fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            loss = outputs.loss
            logits = outputs.logits
        logits = nested_detach(logits)
        labels = torch.tensor([0]*len(logits)).to(logits.device) # dummy labels
        return loss, logits, labels
    
    def prediction_loop(
        self,
        dataloader: torch.utils.data.DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        *args,
        **kwargs,
    ) -> EvalLoopOutput:

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        model = self._wrap_model(self.model, training=False)
        batch_size = dataloader.batch_size
        model.eval()

        losses = []
        preds = None
        label_ids = None

        for step, inputs in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            losses.append(loss.item())

            if preds is None:
                preds = logits
            else:
                preds = torch.cat((preds, logits), dim=0)

            if label_ids is None:
                label_ids = labels
            else:
                label_ids = torch.cat((label_ids, labels), dim=0)

        if self.args.local_rank != -1:
            # If running distributed, gather all results.
            preds = self._nested_gather(preds)
            label_ids = self._nested_gather(label_ids)

        preds = preds.cpu().numpy()
        label_ids = label_ids.cpu().numpy()

        if self.args.training_loss == "triplet":
            accuracy = np.mean(np.argmax(preds, axis=1) == label_ids)
            return EvalLoopOutput(
                predictions=preds, 
                label_ids=label_ids, 
                metrics={"accuracy": accuracy, "loss": np.mean(losses)},
                num_samples=len(label_ids)
            )
        else:
            return EvalLoopOutput(predictions=preds, label_ids=label_ids, metrics={"loss": np.mean(losses)}, num_samples=len(label_ids))