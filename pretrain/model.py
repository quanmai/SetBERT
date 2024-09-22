# Finetune BERT on set dataset, using triplet loss

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List
from torch import Tensor as T
from torch import distributed as dist
from transformers import AutoModel, TrainingArguments, BertModel
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from torch.nn import functional as F

@dataclass
class SetBERTOutput(ModelOutput):
    loss: Optional[T] = None
    logits: Optional[T] = None

class SetBERT(nn.Module):
    def __init__(self, bert, model_args: TrainingArguments, training_args: TrainingArguments):
        super(SetBERT, self).__init__()
        self.bert = bert
        self.fc = nn.Linear(768, model_args.emb_dim) if model_args.project else nn.Identity()
        self.training_loss = training_args.training_loss

    def encode(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask).last_hidden_state
        return self.fc(out[:, 0, :])
    
    def forward(self,
        a_input_ids: T,
        a_attention_mask: T,
        p_input_ids: T,
        p_attention_mask: T,
        n_input_ids: T,
        n_attention_mask: T,
        **kwargs,
    ) -> SetBERTOutput:
        if self.training_loss == "triplet":
            a_out = self.encode(a_input_ids, a_attention_mask) # B x emb_dim
            p_out = self.encode(p_input_ids, p_attention_mask) # B x emb_dim
            n_out = self.encode(n_input_ids, n_attention_mask) # B x emb_dim
            a_out, p_out, n_out = self.gather_tensors(a_out, p_out, n_out)
            # loss = self.triplet_loss(a_out, p_out, n_out)
            d_a_p = torch.cosine_similarity(a_out, p_out)
            d_a_n = torch.cosine_similarity(a_out, n_out)
            loss = torch.relu(1 - d_a_p + d_a_n).mean()
            logits = torch.softmax(torch.stack([d_a_p, d_a_n], dim=1), dim=1)
            return SetBERTOutput(loss=loss, logits=logits)
        elif self.training_loss == "bi_contrastive":
            a_out = self.encode(a_input_ids, a_attention_mask)
            p_out = self.encode(p_input_ids, p_attention_mask)
            n_out = self.encode(n_input_ids, n_attention_mask)
            mp_out = self.encode(kwargs['mp_input_ids'], kwargs['mp_attention_mask'])
            mn_out = self.encode(kwargs['mn_input_ids'], kwargs['mn_attention_mask'])
            a_out, p_out, n_out, mp_out, mn_out = self.gather_tensors(a_out, p_out, n_out, mp_out, mn_out)
            pn_out = torch.cat([p_out, mn_out], dim=0)
            np_out = torch.cat([n_out, mp_out], dim=0)
            scores_pn = torch.einsum("md,nd->mn", a_out, pn_out)
            scores_np = torch.einsum("md,nd->mn", a_out, np_out)
            logits_pn = F.log_softmax(scores_pn, dim=1)
            logits_np = F.log_softmax(-scores_np, dim=1)
            target_pn = torch.arange(scores_pn.size(0), device=scores_pn.device, dtype=torch.long)
            target_np = torch.arange(scores_np.size(0), device=scores_np.device, dtype=torch.long)
            loss = F.nll_loss(logits_pn, target_pn) + F.nll_loss(logits_np, target_np)
            return SetBERTOutput(loss=loss, logits=logits_np)
            # loss = F.nll_loss(logits, target)
        elif self.training_loss == "hybrid": # triplet and contrastive loss
            a_out = self.encode(a_input_ids, a_attention_mask) # B x emb_dim
            p_out = self.encode(p_input_ids, p_attention_mask) # B x emb_dim
            n_out = self.encode(n_input_ids, n_attention_mask) # B x emb_dim
            mp_out = self.encode(kwargs['mp_input_ids'], kwargs['mp_attention_mask'])
            mn_out = self.encode(kwargs['mn_input_ids'], kwargs['mn_attention_mask'])
            a_out, p_out, n_out, mp_out, mn_out = self.gather_tensors(a_out, p_out, n_out, mp_out, mn_out)
            # triplet loss
            d_a_p = torch.cosine_similarity(a_out, p_out)
            d_a_n = torch.cosine_similarity(a_out, n_out)
            # loss_triplet = torch.relu(1 - d_a_p + d_a_n).mean()
            logits = torch.softmax(torch.stack([d_a_p, d_a_n], dim=1), dim=1)

            # contrastive loss
            np_out = torch.cat([n_out, mp_out], dim=0)
            scores = torch.einsum("md,nd->mn", a_out, np_out)
            logits_ = F.log_softmax(-scores, dim=1)
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            loss_contrastive = F.nll_loss(logits_, target)
            return SetBERTOutput(loss=loss_contrastive, logits=logits)
            # return SetBERTOutput(loss=loss_triplet + loss_contrastive, logits=logits)
        else: # contrastive loss
            a_out = self.encode(a_input_ids, a_attention_mask) # B x emb_dim
            p_out = self.encode(p_input_ids, p_attention_mask) # B x emb_dim
            n_out = self.encode(n_input_ids, n_attention_mask) # B x emb_dim
            a_out, p_out, n_out = self.gather_tensors(a_out, p_out, n_out)
            np_out = torch.cat([n_out, p_out], dim=0)
            scores = torch.einsum("md,nd->mn", a_out, np_out)
            logits = F.log_softmax(-scores, dim=1)
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            loss = F.nll_loss(logits, target)
            return SetBERTOutput(loss=loss, logits=logits)

    def triplet_loss(self, a_out, p_out, n_out):
        return torch.relu(1 - torch.cosine_similarity(a_out, p_out) + torch.cosine_similarity(a_out, n_out)).mean()
    
    def nll_loss(self, score: T, target: T) -> T:
        def log_softmin(x, dim=None):
            return F.log_softmax(-x, dim=dim)
        return F.nll_loss(input=log_softmin(score,dim=1),target=target)
    
    def _dist_gather_tensor(self, t: Optional[T]) -> Optional[T]:
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[dist.get_rank()] = t
        # all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
    
    def gather_tensors(self, *t: T):
        t = [torch.cat(self._dist_gather_tensor(tt)) for tt in t]
        return t
    
    @classmethod
    def from_pretrained(cls, model_args: TrainingArguments, training_args: TrainingArguments, *args, **kwargs):
        bert = AutoModel.from_pretrained(model_args.model_name_or_path, *args, **kwargs)
        return cls(bert, model_args, training_args)

    @property
    def device(self):
        return next(self.parameters()).device