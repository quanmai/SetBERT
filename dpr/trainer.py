from transformers.trainer import Trainer, nested_detach, EvalLoopOutput
from torch import nn
from typing import Dict, Union, Any, Tuple, Optional, List
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import AdamW
    
    
def get_linear_scheduler(
    optimizer,
    warmup_steps,
    total_training_steps,
    steps_shift=0,
    last_epoch=-1,
):
    from torch.optim.lr_scheduler import LambdaLR
    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        current_step += steps_shift
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            1e-7,
            float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class DPRTrainer(Trainer):
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
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate, 
            eps=self.args.adam_eps
        )
        return self.optimizer
    
    # def get_scheduler(self, optimizer: torch.optim.Optimizer):
    # def create_scheduler(self, num_training_steps, optimizer: torch.optim.Optimizer):
    #     scheduler = get_linear_scheduler(
    #         optimizer,
    #         self.args.warmup_steps,
    #         num_training_steps,
    #     )
    #     print("HEYYYYAISHKAHSKDJHKDJH")
    #     print(num_training_steps)
    #     return scheduler
    
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
                    output = model(**inputs)
            else:
                output = model(**inputs)
        return output.loss, None, None
    
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
            loss, _, _ = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            losses.append(loss.item())

        final_loss = np.mean(losses)
        return EvalLoopOutput(predictions=preds, label_ids=label_ids, metrics={"loss": final_loss}, num_samples=len(dataloader.dataset))
