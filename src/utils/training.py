import math
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


def get_cosine_lr(epoch: int, max_epochs: int, max_lr: float, min_lr: float) -> float:
    """Cosine annealing learning rate schedule"""
    coeff = 0.5 * (1.0 + math.cos(math.pi * epoch / max_epochs))
    return min_lr + coeff * (max_lr - min_lr)


def get_batch_cfg(cfg, config, sentence_length: int):
    """Generate a batch of CFG data"""
    data, _ = cfg.sample(config.batch_size)
    N = data.shape[0]
    data = data.view(N, sentence_length)
    x = data[:, 0:sentence_length - 1]
    y = data[:, 1:sentence_length].contiguous()
    x, y = x.to(config.device), y.to(config.device)
    return x, y


@torch.no_grad()
def estimate_loss_cfg(model, cfg, config, sentence_length: int, eval_iters: int = 100) -> float:
    """Estimate validation loss on fresh CFG samples"""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch_cfg(cfg, config, sentence_length)
        logits = model(X)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1
        )
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


@torch.no_grad()
def evaluate_cfg_accuracy(model, cfg, context, n_gen: int = 100) -> Tuple[float, np.ndarray]:
    """Evaluate grammatical accuracy on CFG tasks"""
    if isinstance(model, nn.DataParallel):
        model = model.module

    model.eval()
    context_length = context.size()[1]
    sentence_length = np.prod(cfg.T)
    
    gen_sentences = model.generate(
        context, max_new_tokens=sentence_length - context_length, temperature=0.1
    )

    # Compute global accuracy
    gen_sentences = gen_sentences.view([n_gen] + cfg.T).cpu()
    global_acc = cfg.frac_of_gramatically_correct_sentences(gen_sentences)

    # Compute per-level accuracy
    correct_sentences = np.zeros(cfg.L)
    for sentence in gen_sentences:
        _, err = cfg.collapse_and_get_err(sentence)
        for i in range(len(err) - 1, -1, -1):
            if err[i].sum() != 0:
                break
            else:
                correct_sentences[i] += 1

    per_level_acc = np.array(correct_sentences) / n_gen * 100
    model.train()
    return global_acc, per_level_acc


def train_cfg_epoch(model, cfg, config, optimizer, sentence_length: int, batches_per_epoch: int = 50):
    """Train one epoch on CFG data"""
    model.train()
    total_loss = 0.0
    
    for _ in range(batches_per_epoch):
        for _ in range(100):  # Inner loop for more gradient steps
            xb, yb = get_batch_cfg(cfg, config, sentence_length)
            optimizer.zero_grad()
            logits = model(xb)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=-1
            )
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
    
    return total_loss / (batches_per_epoch * 100)


class CFGTrainer:
    """Trainer class for CFG experiments"""
    
    def __init__(self, model, cfg, config, max_lr: float = 6e-4, min_lr: Optional[float] = None):
        self.model = model
        self.cfg = cfg
        self.config = config
        self.max_lr = max_lr
        self.min_lr = min_lr or max_lr / 10
        self.sentence_length = np.prod(cfg.T)
        
        # Setup optimizer
        weight_decay = 1e-1
        beta1, beta2 = 0.9, 0.95
        self.optimizer = model.configure_optimizers(
            weight_decay, max_lr, (beta1, beta2), device_type='cuda'
        )
        
        # Setup evaluation context
        context_length = 8
        self.context = cfg.sample(100)[0].view(100, self.sentence_length)[:, :context_length].to(config.device)

    def train_epoch(self, epoch: int, max_epochs: int, batches_per_epoch: int = 50):
        """Train one epoch with cosine LR schedule"""
        # Update learning rate
        lr = get_cosine_lr(epoch, max_epochs, self.max_lr, self.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Train
        train_loss = train_cfg_epoch(
            self.model, self.cfg, self.config, self.optimizer, 
            self.sentence_length, batches_per_epoch
        )
        
        return train_loss, lr

    def evaluate(self, eval_iters: int = 100, quality_metric_iters: int = 100):
        """Evaluate model performance"""
        val_loss = estimate_loss_cfg(
            self.model, self.cfg, self.config, self.sentence_length, eval_iters
        )
        
        acc, per_level_acc = evaluate_cfg_accuracy(
            self.model, self.cfg, self.context, quality_metric_iters
        )
        
        return val_loss, acc, per_level_acc


def create_cfg_configs():
    """Create standard CFG configurations for experiments"""
    configs = {}
    
    # CFG-7L configuration
    configs['cfg_7l'] = {
        'L': 7,
        'ns': [1, 3, 3, 3, 5, 5, 9, 10],
        'nr': [2, 2, 2, 2, 2, 2, 2],
        'T': [2, 2, 2, 2, 2, 4, 4]
    }
    
    # CFG-9L configuration  
    configs['cfg_9l'] = {
        'L': 9,
        'ns': [1, 3, 3, 3, 5, 5, 5, 9, 9, 10],
        'nr': [2, 2, 2, 2, 2, 2, 2, 2, 2],
        'T': [1, 2, 2, 2, 2, 2, 2, 2, 4]
    }
    
    # Additional depth configurations
    list_ns = [
        [1, 3, 9, 10],
        [1, 3, 5, 9, 10], 
        [1, 3, 3, 5, 9, 10],
        [1, 3, 3, 5, 5, 9, 10],
        [1, 3, 3, 3, 5, 5, 9, 10],
        [1, 3, 3, 3, 5, 5, 5, 9, 10],
        [1, 3, 3, 3, 5, 5, 5, 9, 9, 10],
        [1, 3, 3, 3, 5, 5, 5, 9, 9, 9, 10],
        [1, 3, 3, 3, 3, 5, 5, 5, 9, 9, 9, 10]
    ]
    
    list_T = [
        [8, 8, 4], 
        [2, 8, 8, 4], 
        [2, 2, 8, 4, 4],
        [2, 2, 2, 4, 4, 4], 
        [2, 2, 2, 2, 2, 4, 4], 
        [2, 2, 2, 2, 2, 2, 2, 4],
        [1, 2, 2, 2, 2, 2, 2, 2, 4], 
        [1, 1, 2, 2, 2, 2, 2, 2, 2, 4], 
        [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 4]
    ]
    
    for i, L in enumerate([3, 4, 5, 6, 7, 8, 9, 10, 11]):
        configs[f'cfg_{L}l'] = {
            'L': L,
            'ns': list_ns[i],
            'nr': [2] * L,
            'T': list_T[i]
        }
    
    return configs
