from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os

class SaveModelAndTokenizerCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_save(args, state, control, **kwargs)
        checkpoint_folder = f"{args.output_dir}/{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        # state.model.save_pretrained(checkpoint_folder)
        if state.is_world_process_zero:
            self.tokenizer.save_pretrained(checkpoint_folder)
    
class SaveBestModelCallback(TrainerCallback):
    def __init__(self, save_path):
        self.save_path = save_path
        self.best_eval_loss = float('inf')
    
    def on_evaluate(self, args, state, control, **kwargs):
        eval_loss = kwargs['metrics']['loss']
        best_path = os.path.join(self.save_path, "best_model")
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            
            # Save the query and document encoders separately
            step = state.global_step
            kwargs['model'].bert.save_pretrained(best_path) #, safe_serialization=args.safe_tensors)
            
            # save tokenizer
            if state.is_world_process_zero:
                kwargs['tokenizer'].save_pretrained(best_path)
                print(f"Best eval loss: {eval_loss:.4f} at step {step}. Model saved to {best_path}.")
