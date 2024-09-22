# Finetune BERT on set dataset, using triplet loss
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List, Dict, Union, Any
from torch import Tensor as T
from torch import distributed as dist
from transformers import AutoModel, TrainingArguments, BertModel, AutoModelForSeq2SeqLM
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from torch.nn import functional as F

    
@dataclass
class SetDPROutput(ModelOutput):
    loss: Optional[T] = None
    
class SetDPR(nn.Module):
    def __init__(
        self, 
        query_encoder: nn.Module, 
        document_encoder: nn.Module,
        model_args: TrainingArguments,
    ):
        super(SetDPR, self).__init__()
        self.query_encoder = query_encoder
        self.document_encoder = document_encoder
        # self.query_project = nn.Linear(768, model_args.emb_dim) if model_args.project else nn.Identity()
        # self.document_project = nn.Linear(768, model_args.emb_dim) if model_args.project else nn.Identity()

    def forward(
        self, 
        query_input: dict,
        positive_input: dict,
        negative_input: dict,
    )-> SetDPROutput:
        q_cls = self.encode(self.query_encoder, query_input)
        pos_cls = self.encode(self.document_encoder, positive_input)
        neg_cls = self.encode(self.document_encoder, negative_input)
        if pos_cls.size(0) != q_cls.size(0): # take mean as the document embedding
            num_pos = pos_cls.size(0) // q_cls.size(0)
            pos_cls = pos_cls.view(-1, num_pos, pos_cls.shape[-1]).mean(dim=1, keepdim=False) # B x emb_dim
        q_cls, pos_cls, neg_cls = self.gather_tensors(q_cls, pos_cls, neg_cls)
        doc_cls = torch.cat([pos_cls, neg_cls], dim=0)
        scores = self.compute_similarity(q_cls, doc_cls)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        loss = self.nll_loss(scores, target)
        return SetDPROutput(loss=loss)

    def encode(self, encoder, input: Union[Dict[str, T], Any]):
        if isinstance(encoder, BertModel):
            return encoder(**input).last_hidden_state[:, 0, :]
        else: # t5
            output = encoder(**input)
            return self.mean_pooling(output, input["attention_mask"])
            # return encoder(**input).last_hidden_state.mean(dim=1)
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def nll_loss(self, score: T, target: T) -> T:
        return F.nll_loss(input=F.log_softmax(score,dim=1),target=target)
        
    def compute_similarity(self, q: T, p: T) -> T:
        return torch.einsum("md,nd->mn", q, p)
    
    def _dist_gather_tensor(self, t: Optional[T]) -> Optional[T]:
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[dist.get_rank()] = t
        return all_tensors
    
    def gather_tensors(self, *t: T):
        t = [torch.cat(self._dist_gather_tensor(tt)) for tt in t]
        return t

    @classmethod
    def from_pretrained(
        cls,
        model_args: TrainingArguments,
        *args,
        **kwargs,
    ) -> nn.Module:
        if "t5" in model_args.query_encoder_model_name_or_path:
            query_encoder = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.query_encoder_model_name_or_path,
                use_safetensors=True,
                *args, **kwargs
            ).encoder
        else:
            query_encoder = AutoModel.from_pretrained(
                model_args.query_encoder_model_name_or_path, 
                use_safetensors=True, #model_args.query_encoder_model_name_or_path.endswith("safetensors"), 
                *args, **kwargs
            )
        if "t5" in model_args.document_encoder_model_name_or_path:
            passage_encoder = AutoModelForSeq2SeqLM.from_pretrained(model_args.document_encoder_model_name_or_path, *args, **kwargs).encoder
        else:
            passage_encoder = AutoModel.from_pretrained(
                model_args.document_encoder_model_name_or_path, 
                use_safetensors=True,
                *args, 
                **kwargs
            )
        model = cls(query_encoder, passage_encoder, model_args)
        return model
        