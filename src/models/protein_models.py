import torch
import torch.nn as nn
import math
from dataclasses import dataclass, field
from jaxtyping import jaxtyped, Float
from torch import Tensor


# Protein vocabulary
periodic_table_elements = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'
]

amino_acids = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
]


@dataclass
class ProtConfig:
    atom_vocab_size: int = len(periodic_table_elements)
    residue_vocab_size: int = len(amino_acids)
    atom_block_size: int = 2000
    residue_block_size: int = 100
    n_layer: int = 6
    n_head: int = 6
    head_size: int = 64
    embed_dim_total: int = field(init=False)
    dropout: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
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
    """Multi-head self-attention for proteins"""
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=config.embed_dim_total,
            num_heads=config.n_head,
            dropout=config.dropout,
            batch_first=True,
        )

    def forward(self, x):
        mask = torch.tril(torch.ones((x.size(1), x.size(1)), dtype=torch.bool, device=x.device))
        mask = mask.unsqueeze(0).repeat(self.n_head * x.size(0), 1, 1).float()
        mask = mask.masked_fill(mask == 0, float("-inf"))
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=mask)
        return attn_output


class MLP(nn.Module):
    def __init__(self, config: ProtConfig):
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
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.embed_dim_total, bias=config.bias)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config.embed_dim_total, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: ProtConfig, part: str):
        super().__init__()
        self.config = config
        self.vocab_size = config.atom_vocab_size if part == "atom" else config.residue_vocab_size
        self.block_size = config.atom_block_size if part == "atom" else config.residue_block_size
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.vocab_size, config.embed_dim_total),
            wpe=nn.Embedding(self.block_size, config.embed_dim_total),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.embed_dim_total, bias=config.bias),
        ))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

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
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(x)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x


class ProteinHierarchicalTransformer(nn.Module):
    """Hierarchical Transformer for protein structure prediction"""
    def __init__(self, config: ProtConfig):
        super().__init__()
        self.config = config
        
        self.atom_transformer = Transformer(config, "atom")
        self.res_transformer = Transformer(config, "res")
        self.lm_head = nn.Linear(config.embed_dim_total, 3, bias=False)
        self.dim_reducer = nn.Sequential(
            nn.Linear(2 * config.embed_dim_total, config.embed_dim_total), 
            nn.GELU()
        )

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("Protein HT parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.atom_transformer.transformer.wpe.weight.numel()
            n_params -= self.res_transformer.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, atom_seq, residue_seq, counts):
        b, t = residue_seq.shape
        assert t <= self.config.residue_block_size
        assert atom_seq.size(0) <= self.config.atom_block_size

        atom_out = self.atom_transformer(atom_seq)
        residue_out = self.res_transformer(residue_seq)
        
        # Extend residue embeddings to match atom sequence length
        residue_out_extended = torch.repeat_interleave(
            residue_out.squeeze(0), counts.squeeze(0), dim=0
        ).unsqueeze(0)
        
        out = self.dim_reducer(torch.cat((atom_out, residue_out_extended), dim=-1))
        coordinates = self.lm_head(out)
        return coordinates

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer


class RegularTransformer(nn.Module):
    """Regular Transformer for protein structure prediction"""
    def __init__(self, config: ProtConfig):
        super().__init__()
        self.config = config
        
        self.atom_transformer = Transformer(config, "atom")
        self.lm_head = nn.Linear(config.embed_dim_total, 3, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("Protein RT parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.atom_transformer.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, atom_seq, residue_seq, counts):
        assert atom_seq.size(0) <= self.config.atom_block_size
        atom_out = self.atom_transformer(atom_seq)
        coordinates = self.lm_head(atom_out)
        return coordinates

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer


class KabschMSELoss(nn.Module):
    """Kabsch-aligned MSE loss for protein structure prediction"""
    def __init__(self):
        super(KabschMSELoss, self).__init__()

    @jaxtyped
    def forward(
        self,
        model_output: Float[Tensor, "b n 3"],
        ground_truth: Float[Tensor, "b n 3"],
    ) -> Float[Tensor, ""]:
        model_output, ground_truth = model_output.squeeze(0), ground_truth.squeeze(0)
        mobile_aligned = kabsch_alignment(mobile_pts=ground_truth, target_pts=model_output)
        squared_errs = (model_output - mobile_aligned).pow(2)
        return squared_errs.mean()


@torch.no_grad()
def kabsch_alignment(mobile_pts, target_pts):
    """Kabsch algorithm for optimal alignment of two point sets"""
    # Center the coordinates
    P_centered = mobile_pts - mobile_pts.mean(dim=0)
    target_pts_centered = target_pts - target_pts.mean(dim=0)

    # Compute the covariance matrix
    C = torch.mm(P_centered.T, target_pts_centered)

    # Perform SVD
    V, S, Wt = torch.svd(C)

    # Compute the rotation matrix
    d = torch.sign(torch.det(torch.mm(Wt.T, V.T)))
    D = torch.diag(torch.tensor([1, 1, d], dtype=mobile_pts.dtype, device=mobile_pts.device))
    rotation_matrix = torch.mm(Wt.T, torch.mm(D, V.T))
    
    # Apply rotation and translation
    P_rotated = torch.mm(P_centered, rotation_matrix)
    translation_vector = target_pts.mean(dim=0) - P_rotated.mean(dim=0)
    mobile_aligned = P_rotated + translation_vector

    return mobile_aligned
