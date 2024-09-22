from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os

class SaveModelAndTokenizerCallback(TrainerCallback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # super().on_save(args, state, control, **kwargs)
        # save only when the global step is larger than the 
        checkpoint_folder = f"{self.save_path}/{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        query_encoder_save_path = os.path.join(checkpoint_folder, "query_encoder")
        doc_encoder_save_path = os.path.join(checkpoint_folder, "doc_encoder")
        kwargs['model'].query_encoder.save_pretrained(query_encoder_save_path)
        kwargs['model'].document_encoder.save_pretrained(doc_encoder_save_path)
        if state.is_world_process_zero:
            kwargs['tokenizer'].save_pretrained(query_encoder_save_path)
            kwargs['tokenizer'].save_pretrained(doc_encoder_save_path)
    
class SaveBiEncoderCallback(TrainerCallback):
    def __init__(self, save_path, save_tokenizer=True):
        self.save_path = save_path
        self.best_eval_loss = float('inf')
        self.save_tokenizer = save_tokenizer
    
    def on_evaluate(self, args, state, control, **kwargs):
        eval_loss = kwargs['metrics']['loss']
        # only evaluate when current step is bigger than 200            
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            
            # Save the query and document encoders separately
            step = state.global_step
            query_encoder_save_path = os.path.join(self.save_path, "query_encoder")
            doc_encoder_save_path = os.path.join(self.save_path, "doc_encoder")
            
            kwargs['model'].query_encoder.save_pretrained(query_encoder_save_path) #, safe_serialization=args.safe_tensors)
            kwargs['model'].document_encoder.save_pretrained(doc_encoder_save_path) #, safe_serialization=args.safe_tensors)
            
            # save tokenizer
            if state.is_world_process_zero and self.save_tokenizer:
                kwargs['tokenizer'].save_pretrained(query_encoder_save_path)
                kwargs['tokenizer'].save_pretrained(doc_encoder_save_path)
            if state.is_world_process_zero:
                print(f"Best eval loss: {eval_loss:.4f} at step {step}. Encoders saved to {self.save_path}.")
