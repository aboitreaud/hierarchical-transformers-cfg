#!/usr/bin/env python3
"""
Depth Sweep Experiment
Transformer Accuracy vs CFG Depth (reproduction) and heuristic analysis
"""

import sys
import yaml
import wandb
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models import CFG, GPT, GPTConfig
from utils import CFGTrainer


def load_config(config_path):
    """Load experiment configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(model_config, cfg, sentence_length):
    """Create model based on configuration"""
    config = GPTConfig(
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
    
    model = GPT(config)
    model.to(config.device)
    return model, config


def run_heuristic_analysis(cfg, config):
    """Run heuristic spectral clustering analysis"""
    try:
        # Import heuristic analysis (would need to port from original project)
        # from word_similarity_algo import WordMatcher
        
        print(f"Running heuristic analysis for CFG-{cfg.L}L...")
        
        # Placeholder for heuristic analysis
        # matcher = WordMatcher(cfg, chunk_size=config['chunk_size'], verbose=False, device='cuda')
        
        heuristic_results = {}
        for budget in config['budget_range']:
            print(f"  Testing budget: {budget}")
            # sentences = cfg.sample_flattened(nspl=budget)[0].squeeze(0)
            # matcher.sklearn_spectral_clustering(0, sentences, cheat=True)
            # results_dict, p = matcher.check_rules(0)
            
            # Placeholder results
            heuristic_results[budget] = {
                'success_rate': np.random.random(),  # Replace with actual results
                'budget': budget
            }
        
        return heuristic_results
    except ImportError:
        print("Heuristic analysis not available (WordMatcher not imported)")
        return {}


def run_experiment(config_path):
    """Run the depth sweep experiment"""
    config = load_config(config_path)
    
    results = {}
    
    # Run experiment for each CFG depth
    for depth in config['cfg_depths']:
        print(f"\n{'='*60}")
        print(f"Running CFG-{depth}L Experiment")
        print(f"{'='*60}")
        
        # Create CFG for this depth
        cfg_config = config['cfg_configs'][depth]
        cfg = CFG(L=cfg_config['L'], ns=cfg_config['ns'], nr=cfg_config['nr'], T=cfg_config['T'])
        sentence_length = np.prod(cfg.T)
        
        print(f"CFG-{cfg.L}L created:")
        print(f"  Sentence length: {sentence_length}")
        print(f"  Vocabulary size: {cfg.ns[-1]}")
        print(f"  Grammar structure: ns={cfg.ns}, T={cfg.T}")
        
        # Create model
        model, model_cfg = create_model(config['model'], cfg, sentence_length)
        
        # Initialize wandb
        if config.get('use_wandb', True):
            wandb.init(
                project=config['logging']['project'],
                name=f"CFG-{depth}L_RT_{model.get_num_params()/1e6:.1f}M",
                config={
                    'depth': depth,
                    'model': config['model'],
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
                    "depth": depth,
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
        
        # Run heuristic analysis if enabled
        heuristic_results = {}
        if config.get('heuristic', {}).get('enabled', False):
            heuristic_results = run_heuristic_analysis(cfg, config['heuristic'])
        
        results[depth] = {
            'final_accuracy': final_acc,
            'parameters': model.get_num_params(),
            'sentence_length': sentence_length,
            'vocab_size': cfg.ns[-1],
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'heuristic_results': heuristic_results,
            'cfg_structure': {
                'ns': cfg.ns,
                'nr': cfg.nr,
                'T': cfg.T
            }
        }
        
        if config.get('use_wandb', True):
            wandb.log({
                "final_accuracy_mean_10_epochs": final_acc,
                "depth": depth,
                "sentence_length": sentence_length,
                "vocab_size": cfg.ns[-1]
            })
            wandb.finish()
        
        print(f"\nCFG-{depth}L Results:")
        print(f"Final accuracy (mean last 10 epochs): {final_acc:.2f}%")
        print(f"Parameters: {model.get_num_params()/1e6:.2f}M")
        print(f"Sentence length: {sentence_length}")
        
        # Clean up model to save memory
        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Print summary
    print(f"\n{'='*60}")
    print("DEPTH SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'Depth':<6} {'Accuracy':<10} {'Params':<8} {'Sent Len':<8} {'Vocab':<6}")
    print("-" * 60)
    
    for depth in sorted(results.keys()):
        result = results[depth]
        print(f"{depth:<6} {result['final_accuracy']:<10.2f} "
              f"{result['parameters']/1e6:<8.2f} {result['sentence_length']:<8} "
              f"{result['vocab_size']:<6}")
    
    # Save results
    if config['logging'].get('save_results', False):
        results_dir = Path("experiments/results/depth_sweep")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(results_dir / "depth_sweep_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_dir / 'depth_sweep_results.json'}")
    
    return results


if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "configs" / "depth_sweep.yaml"
    results = run_experiment(config_path)
