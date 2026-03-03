import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from typing import Tuple, Optional, Dict, List, Union
from loguru import logger


def mean_pooling(token_embeddings, attention_mask=None):
    if attention_mask is None:
        return token_embeddings.mean(1)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def cosine_similarity(x, y):
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    return torch.matmul(x_norm, y_norm.t())


def contrastive_loss_with_masked_negatives(X, Y, tau=0.07, margin=0.1):
    B, d = X.shape
    
    sim_pos = F.cosine_similarity(X, Y, dim=-1)
    exp_pos = torch.exp(sim_pos / tau)
    
    all_sim_qq = cosine_similarity(X, X)
    all_sim_qy = cosine_similarity(X, Y)
    
    mask = torch.eye(B, dtype=torch.bool, device=X.device)
    neg_mask = ~mask
    
    sim_pos_expanded = sim_pos.unsqueeze(1).expand(-1, B)
    
    m_ij = torch.ones_like(all_sim_qy, dtype=torch.float32)
    m_ij[neg_mask] = (all_sim_qy[neg_mask] <= sim_pos_expanded[neg_mask] + margin).float()
    
    all_sim_qy_neg = all_sim_qy[neg_mask].view(B, -1)
    m_ij_neg = m_ij[neg_mask].view(B, -1)
    
    exp_neg = torch.exp(all_sim_qy_neg / tau) * m_ij_neg
    
    Z = exp_pos + exp_neg.sum(dim=1)
    
    loss = -torch.log(exp_pos / (Z + 1e-8)).mean()
    return loss


class SimpleLowDimProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)


def _create_bidirectional_mask(attention_mask):
    B, L = attention_mask.shape
    device = attention_mask.device
    
    valid_positions = torch.einsum('bi,bj->bij', attention_mask, attention_mask).float()
    
    valid_positions = valid_positions.unsqueeze(1)  # [B, 1, L, L]
    
    bidirectional_mask = torch.where(
        valid_positions.bool(),
        torch.tensor(0.0, dtype=torch.float32, device=device),
        torch.tensor(torch.finfo(torch.float32).min, dtype=torch.float32, device=device)
    )
    
    return bidirectional_mask



class FindGGSM(nn.Module):
    
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        
        logger.info(f"📥 Loading LLM from {config.llm_model_name}")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True
        )
        
        self.low_dim_projector = SimpleLowDimProjector(
            config.hidden_size, 512, 128
        )
        
        max_len = getattr(config, "max_position_embeddings", 2048)
        self.register_buffer("future_importance", torch.zeros(max_len, max_len))
        self.register_buffer("frozen_importance_weights", torch.zeros(max_len, max_len))
        
        self.ema_decay = config.ema_decay
        self._grad_buffers = {}
        self.warmup_steps = config.warmup_steps
        self.total_training_steps = config.total_training_steps
        self._importance_frozen = False
    
    def _save_grad_hook(self, name):
        def hook(grad):
            self._grad_buffers[name] = grad.clone()
        return hook
    
    @torch.no_grad()
    def estimate_future_importance_from_grads(self, input_ids, attention_mask, current_step):
        if 'hidden_state' not in self._grad_buffers:
            return
        
        grad = self._grad_buffers['hidden_state']
        grad_norm = grad.norm(dim=-1)  # [B, L]
        B, L = grad_norm.shape
        
        upper_tri = torch.triu(torch.ones(L, L, device=grad_norm.device), diagonal=1)
        grad_j = grad_norm.unsqueeze(1).expand(-1, L, L)
        delta_batch = grad_j * upper_tri
        
        seq_lens = attention_mask.sum(dim=1)
        valid_mask = torch.arange(L, device=grad_norm.device)[None, :] < seq_lens[:, None]
        valid_mask = valid_mask[:, None, :] & valid_mask[:, :, None]
        delta_batch = delta_batch * valid_mask
        delta = delta_batch.sum(dim=0)
        
        if delta.abs().sum() > 0:
            self.future_importance[:L, :L] = (
                self.ema_decay * self.future_importance[:L, :L] +
                (1 - self.ema_decay) * delta
            )
        
        self._grad_buffers.clear()
        
        if not self._importance_frozen and current_step == self.warmup_steps - 1:
            frozen_weights = torch.sigmoid(self.future_importance[:L, :L])
            self.frozen_importance_weights[:L, :L].copy_(frozen_weights)
            self._importance_frozen = True
            logger.info(f"Frozen importance weights at step {current_step}")
    
    def _get_soft_causal_bias(self, seq_len: int, device: torch.device, current_step: int, dtype=torch.float32) -> Tuple[torch.Tensor, str]:
        if seq_len <= 1:
            return torch.zeros(seq_len, seq_len, dtype=dtype, device=device), "none"
        
        total_sched_steps = max(1, self.total_training_steps - self.warmup_steps)
        
        if current_step >= self.warmup_steps + total_sched_steps:
            return torch.zeros(seq_len, seq_len, dtype=dtype, device=device), "full_bi"
        
        if current_step < self.warmup_steps:
            imp = self.future_importance[:seq_len, :seq_len]
            weights = torch.sigmoid(imp)
            log_weights = torch.log(weights + 1e-8)
            
            bias = torch.full((seq_len, seq_len), float('-1e6'), dtype=dtype, device=device)
            causal_mask = ~torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
            bias[causal_mask] = 0.0
            upper_tri_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
            bias[upper_tri_mask] = log_weights[upper_tri_mask].to(dtype)
            return bias, "warmup"
        
        else:
  
            alpha = min((current_step - self.warmup_steps) / total_sched_steps, 1.0)
            
            if alpha >= 1.0:
                return torch.zeros(seq_len, seq_len, dtype=dtype, device=device), "scheduler_full"
            
            bias = torch.full((seq_len, seq_len), float('-1e6'), dtype=dtype, device=device)
            causal_mask = ~torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
            bias[causal_mask] = 0.0
            upper_tri_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
            
            if self.frozen_importance_weights.abs().sum() > 0:
                if seq_len <= self.frozen_importance_weights.size(0):
                    base_w_upper = self.frozen_importance_weights[:seq_len, :seq_len][upper_tri_mask]
                    future_weights = (1.0 - alpha) * base_w_upper + alpha
                    # future_weights = torch.clamp(future_weights, 1e-8, 1.0)
                    bias[upper_tri_mask] = torch.log(future_weights).to(dtype)
                else:
                    future_log_weight = math.log(alpha)
                    bias[upper_tri_mask] = future_log_weight
            else:
                future_log_weight = math.log(alpha)
                bias[upper_tri_mask] = future_log_weight
            
            return bias, "scheduler"

    def forward(
        self,
        query_input_ids: torch.LongTensor,
        query_attention_mask: torch.Tensor,
        answer_input_ids: torch.LongTensor = None,
        answer_attention_mask: torch.Tensor = None,
        current_step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_embeds = self.llm_model.get_input_embeddings()(query_input_ids)
        B, L = query_input_ids.shape
        embed_dtype = query_embeds.dtype
        mode=None
        if self.training:
            soft_bias_query, mode = self._get_soft_causal_bias(
                L, query_input_ids.device, current_step, dtype=embed_dtype
            )
            attn_mask_query_4d = soft_bias_query.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).contiguous()
            use_cache_local = False
            
            upper_tri_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=query_input_ids.device), diagonal=1)
            if upper_tri_mask.any():
                future_weights = torch.exp(soft_bias_query[upper_tri_mask])
                avg_weight = future_weights.mean().item()
            else:
                avg_weight = 0.0
        else:
            attn_mask_query_4d = None
            use_cache_local = False
            avg_weight = 0.0
        
        query_outputs = self.llm_model.model(
            input_ids=None,
            attention_mask=attn_mask_query_4d,
            inputs_embeds=query_embeds,
            use_cache=use_cache_local,
            output_hidden_states=True,
            return_dict=True,
        )
        query_hidden = query_outputs.hidden_states[-1]  # [B, L, D]
        
        if self.training:
            query_hidden.retain_grad()
            query_hidden.register_hook(self._save_grad_hook('hidden_state'))
        
        query_emb = mean_pooling(query_hidden, query_attention_mask)  # [B, D]
        query_emb_low = self.low_dim_projector(query_emb)  # [B, 128]
        
        if answer_input_ids is not None:
            answer_embeds = self.llm_model.get_input_embeddings()(answer_input_ids)
            B_ans, L_ans = answer_input_ids.shape
            
            soft_bias_answer, _ = self._get_soft_causal_bias(L_ans, query_input_ids.device,current_step=current_step)
            attn_mask_answer_4d = soft_bias_answer.unsqueeze(0).unsqueeze(0).expand(B_ans, 1, -1, -1).contiguous()
            
            answer_outputs = self.llm_model.model(
                input_ids=None,
                attention_mask=attn_mask_answer_4d,
                inputs_embeds=answer_embeds,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            answer_hidden = answer_outputs.hidden_states[-1]
            
            answer_emb = mean_pooling(answer_hidden, answer_attention_mask)
            answer_emb_low = self.low_dim_projector(answer_emb)
            
            loss_contra = contrastive_loss_with_masked_negatives(
                query_emb_low, answer_emb_low, 
                tau=self.config.logit_scale,
                margin=self.config.margin
            )
        else:
            loss_contra = contrastive_loss_with_masked_negatives(
                query_emb_low, query_emb_low, 
                tau=self.config.logit_scale,
                margin=self.config.margin
            )
        
        return loss_contra, avg_weight,mode

    def evaluate(
        self,
        query_input_ids: torch.LongTensor,
        query_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        query_embeds = self.llm_model.get_input_embeddings()(query_input_ids)
        
        B, L = query_input_ids.shape
        attn_mask = torch.ones(B, 1, L, L, dtype=torch.float32, device=query_input_ids.device)
        
        query_outputs = self.llm_model.model(
            input_ids=None,
            attention_mask=attn_mask,
            inputs_embeds=query_embeds,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        query_hidden = query_outputs.last_hidden_state
        
        query_emb = mean_pooling(query_hidden, query_attention_mask)
        
        query_emb_low = self.low_dim_projector(query_emb)
        query_emb_norm = F.normalize(query_emb_low, p=2, dim=1)
        
        return query_emb_norm
