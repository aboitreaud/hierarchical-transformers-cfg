import numpy as np
import torch
import torch.nn as nn
import math
from dataclasses import dataclass, field
from .cfg import CFG


@dataclass
class GPTConfig:
    n_transformers: int = 1
    vocab_size: int = None
    block_size: int = 256
    batch_size: int = 256
    n_layer: int = 12
    n_head: int = 12
    head_size: int = 64
    embed_dim_total: int = field(init=False)
    dropout: float = 0.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    bias: bool = False

    def __post_init__(self):
        self.embed_dim_total = self.n_head * self.head_size


class LayerNorm(nn.Module):
    """LayerNorm with optional bias"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return nn.functional.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=config.embed_dim_total,
            num_heads=config.n_head,
            dropout=config.dropout,
            batch_first=True
        )

    def forward(self, x):
        mask = torch.tril(torch.ones((x.size(1), x.size(1)), dtype=torch.bool, device=x.device))
        mask = mask.unsqueeze(0).repeat(self.n_head * x.size(0), 1, 1).float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=mask)
        return attn_output


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.embed_dim_total, 4 * config.embed_dim_total, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.embed_dim_total, config.embed_dim_total, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.embed_dim_total, bias=config.bias)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config.embed_dim_total, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Regular Transformer (RT) model"""
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.embed_dim_total),
            wpe=nn.Embedding(config.block_size, config.embed_dim_total),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.embed_dim_total, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.embed_dim_total, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("RT parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class ParentEmbedder(nn.Module):
    """Hierarchical embedding for parent-child relationships"""
    def __init__(self, config: GPTConfig, children: int):
        super(ParentEmbedder, self).__init__()
        self.dmodel = config.embed_dim_total
        self.c = children
        self.mlp = nn.Linear(
            in_features=config.embed_dim_total * self.c,
            out_features=config.embed_dim_total
        )
        self.mlp.to(config.device)

    def forward(self, x):
        B, T, embed_dim_total = x.shape
        complete_parents = T // self.c
        x = x[:, :complete_parents * self.c, :]
        x = x.view(B, complete_parents, self.c * embed_dim_total)
        output = self.mlp(x)
        return output


def custom_prod(arr):
    return np.prod(arr) if len(arr) > 0 else 0


class Transformer(nn.Module):
    """Individual transformer for hierarchical model"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wpe=nn.Embedding(config.block_size, config.embed_dim_total),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.embed_dim_total, bias=config.bias),
        ))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        device = x.device
        b, t, n_embd = x.size()
        assert t <= self.config.block_size, (f"Cannot forward sequence of length {t}, "
                                             f"block size is only {self.config.block_size}")
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(x + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x


class HierarchicalTransformer(nn.Module):
    """Hierarchical Transformer (HT) model"""
    def __init__(self, config: GPTConfig, cfg: CFG):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.c = cfg.T[::-1]  # children per node from leaves to root

        children_embd = nn.Embedding(config.vocab_size, config.embed_dim_total)
        
        self.transformers = nn.ModuleList([
            Transformer(config) for _ in range(config.n_transformers)
        ])
        
        # 1 parent embedder per transformer except for the children level transformer
        parent_embedders = [
            ParentEmbedder(config=config, children=self.c[level]) 
            for level in range(self.config.n_transformers - 1)
        ]

        # Eg: for 3 transformers, we need one dimension reducer (2D, D) and one (3D, D)
        self.dimension_reducer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(i*config.embed_dim_total, config.embed_dim_total),
                nn.GELU()
            ) for i in range(2, self.config.n_transformers + 1)
        ])
        
        self.lm_head = nn.Linear(config.embed_dim_total, config.vocab_size, bias=False)
        children_embd.weight = self.lm_head.weight
        self.embedders = [children_embd] + parent_embedders

        print("HT parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        # x is 2D (B, T), not embedded yet
        # The first module in self.embedders transforms it into token embeddings of shape (b, t, n_embd)
        B, T = x.shape
        transformer_outputs = []
        
        for l in range(self.config.n_transformers):
            if T - custom_prod(self.c[:l]) > 0:
                # For l=0, it's the children embedding first and then the parent embedders that are used
                x = self.embedders[l](x)
                # Forward the transformer blocks
                x = self.transformers[l](x)
                # Concat parent and children outputs
                # Since a child is only aware of the predecessors of its parent, the first c children don't know any parent
                # Last parent may be incomplete, ie not having c children.
                # Discard as many embeddings as the number of missing children for that last parent
                if l > 0:
                    x = torch.repeat_interleave(
                        x, np.prod(self.c[:l]), dim=-2)  # (B, T//prod(c)*prod(c), D)
                    # Discard the first nodes that don't know about their ancestor at level l, ie prod(c) first ones
                    target_seq_len = T - np.prod(self.c[:l])
                    x = x[:, :target_seq_len, :]
                transformer_outputs.append(x)
        
        out = [transformer_outputs[0][:, :self.c[0], :]]
        for l in range(1, len(transformer_outputs)-1):
            cat = torch.cat([
                t[:, np.prod(self.c[:l]) - custom_prod(self.c[:idx]):np.prod(self.c[:l+1]) - custom_prod(self.c[:idx]), :] 
                for idx, t in enumerate(transformer_outputs[:l+1])
            ], dim=-1)
            out.append(self.dimension_reducer[l-1](cat))
        
        # for the last iter, take full length of the sentences
        common_length = transformer_outputs[-1].size(1)
        cat = torch.cat([t[:, -common_length:, :] for t in transformer_outputs], dim=-1)
        out.append(self.dimension_reducer[len(transformer_outputs)-2](cat))

        logits = self.lm_head(torch.cat(out, dim=-2))
        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
