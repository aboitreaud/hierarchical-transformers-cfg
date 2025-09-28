#!/usr/bin/env python3
"""
CFG-7L RT vs HT Experiment
Runs RT and HT on CFG-7L with cosine LR 6e-4→6e-5, 100 epochs, fresh 5k sentences/epoch
"""

import sys
import yaml
import wandb
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models import CFG, GPT, HierarchicalTransformer, GPTConfig
from utils import CFGTrainer


def load_config(config_path):
    """Load experiment configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(model_config, cfg, sentence_length):
    """Create model based on configuration"""
    config = GPTConfig(
        n_transformers=model_config.get('n_transformers', 1),
        vocab_size=cfg.ns[-1],
        block_size=sentence_length - 1,
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        head_size=model_config['head_size'],
        dropout=model_config['dropout'],
        bias=model_config['bias'],
        batch_size=100,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    if model_config['type'] == 'GPT':
        model = GPT(config)
    elif model_config['type'] == 'HierarchicalTransformer':
        model = HierarchicalTransformer(config, cfg)
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    model.to(config.device)
    return model, config


def run_experiment(config_path):
    """Run the CFG-7L RT vs HT experiment"""
    config = load_config(config_path)
    
    # Create CFG
    cfg_config = config['cfg']
    cfg = CFG(L=cfg_config['L'], ns=cfg_config['ns'], nr=cfg_config['nr'], T=cfg_config['T'])
    sentence_length = np.prod(cfg.T)
    
    print(f"CFG-{cfg.L}L created with sentence length: {sentence_length}")
    print(f"Vocabulary size: {cfg.ns[-1]}")
    
    # Run experiments for both models
    results = {}
    
    for model_name, model_config in config['models'].items():
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*50}")
        
        # Create model
        model, model_cfg = create_model(model_config, cfg, sentence_length)
        
        # Initialize wandb
        if config.get('use_wandb', True):
            wandb.init(
                project=config['logging']['project'],
                name=f"{model_name}_{cfg.L}L_{model.get_num_params()/1e6:.1f}M",
                config={
                    'model': model_config,
                    'cfg': cfg_config,
                    'training': config['training'],
                    'sentence_length': sentence_length,
                    'parameters': model.get_num_params()
                }
            )
            wandb.watch(model, log='all')
        
        # Create trainer
        trainer = CFGTrainer(
            model=model,
            cfg=cfg,
            config=model_cfg,
            max_lr=config['training']['max_lr'],
            min_lr=config['training']['min_lr']
        )
        
        # Training loop
        training_config = config['training']
        num_epochs = training_config['num_epochs']
        batches_per_epoch = training_config['batches_per_epoch']
        eval_interval = training_config['eval_interval']
        
        running_acc = []
        
        for epoch in range(num_epochs):
            # Train epoch
            train_loss, lr = trainer.train_epoch(epoch, num_epochs, batches_per_epoch)
            
            # Evaluate
            if epoch % eval_interval == 0 or epoch == num_epochs - 1:
                val_loss, acc, per_level_acc = trainer.evaluate(
                    eval_iters=training_config['eval_iters'],
                    quality_metric_iters=training_config['quality_metric_iters']
                )
                
                running_acc.append(acc * 100)
                
                # Log metrics
                log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "accuracy": acc * 100,
                    "learning_rate": lr,
                    "sentences_seen": (epoch + 1) * batches_per_epoch * model_cfg.batch_size
                }
                
                # Add per-level accuracies
                for i, level_acc in enumerate(per_level_acc):
                    log_dict[f'accuracy_level_{i}'] = level_acc
                
                if config.get('use_wandb', True):
                    wandb.log(log_dict)
                
                if epoch % (eval_interval * 2) == 0:
                    print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                          f"accuracy={acc*100:.2f}%, lr={lr:.2e}")
        
        # Final metrics
        final_acc = np.mean(running_acc[-10:]) if len(running_acc) >= 10 else np.mean(running_acc)
        results[model_name] = {
            'final_accuracy': final_acc,
            'parameters': model.get_num_params(),
            'final_train_loss': train_loss,
            'final_val_loss': val_loss
        }
        
        if config.get('use_wandb', True):
            wandb.log({"final_accuracy_mean_10_epochs": final_acc})
            wandb.finish()
        
        print(f"\n{model_name.upper()} Results:")
        print(f"Final accuracy (mean last 10 epochs): {final_acc:.2f}%")
        print(f"Parameters: {model.get_num_params()/1e6:.2f}M")
        
        # Save checkpoint if requested
        if config['logging'].get('save_checkpoints', False):
            checkpoint_dir = Path("experiments/results/cfg_7l_rt_ht/checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / f"{model_name}_final.pt")
    
    # Print comparison
    print(f"\n{'='*50}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*50}")
    for model_name, result in results.items():
        print(f"{model_name.upper()}: {result['final_accuracy']:.2f}% accuracy, "
              f"{result['parameters']/1e6:.2f}M params")
    
    return results


if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "configs" / "cfg_7l_rt_ht.yaml"
    results = run_experiment(config_path)
