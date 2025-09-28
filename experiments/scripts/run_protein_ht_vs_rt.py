#!/usr/bin/env python3
"""
Protein HT vs RT with Kabsch-MSE Experiment
Compare Hierarchical vs Regular Transformer on protein structure prediction
"""

import sys
import yaml
import wandb
import torch
import torch.nn as nn
import pickle
import math
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models import (
    ProteinHierarchicalTransformer, 
    RegularTransformer, 
    ProtConfig, 
    KabschMSELoss
)


def load_config(config_path):
    """Load experiment configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_cosine_lr(epoch: int, max_epochs: int, max_lr: float, min_lr: float) -> float:
    """Cosine annealing learning rate schedule"""
    coeff = 0.5 * (1.0 + math.cos(math.pi * epoch / max_epochs))
    return min_lr + coeff * (max_lr - min_lr)


def create_model(model_config):
    """Create protein model based on configuration"""
    config = ProtConfig(
        atom_vocab_size=model_config['atom_vocab_size'],
        residue_vocab_size=model_config['residue_vocab_size'],
        atom_block_size=model_config['atom_block_size'],
        residue_block_size=model_config['residue_block_size'],
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        head_size=model_config['head_size'],
        dropout=model_config['dropout'],
        bias=model_config['bias'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    if model_config['type'] == 'ProteinHierarchicalTransformer':
        model = ProteinHierarchicalTransformer(config)
    elif model_config['type'] == 'RegularTransformer':
        model = RegularTransformer(config)
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    model.to(config.device)
    return model, config


def load_protein_dataset(dataset_path, train_test_ratio=0.8):
    """Load and split protein dataset"""
    try:
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        
        torch.manual_seed(0)  # For reproducible splits
        train_size = int(train_test_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        return train_dataset, val_dataset
    except FileNotFoundError:
        print(f"Error: Protein dataset not found at {dataset_path}")
        print("Please ensure the protein dataset is available or create it using the data preparation script.")
        return None, None


def run_experiment(config_path):
    """Run the protein HT vs RT experiment"""
    config = load_config(config_path)
    
    # Load dataset
    dataset_path = config['data_paths']['protein_dataset']
    train_dataset, val_dataset = load_protein_dataset(
        dataset_path, 
        config['dataset']['train_test_ratio']
    )
    
    if train_dataset is None:
        print("Cannot proceed without dataset. Exiting.")
        return {}
    
    print(f"Dataset loaded: {len(train_dataset)} training, {len(val_dataset)} validation samples")
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['dataset']['batch_size'], 
        shuffle=config['dataset']['shuffle']
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config['dataset']['batch_size'], 
        shuffle=False
    )
    
    # Run experiments for both models
    results = {}
    
    for model_name, model_config in config['models'].items():
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*50}")
        
        # Create model
        model, model_cfg = create_model(model_config)
        
        # Setup loss and optimizer
        criterion = KabschMSELoss()
        
        # Training parameters
        training_config = config['training']
        num_epochs = training_config['num_epochs']
        max_lr = training_config['max_lr']
        min_lr = training_config['min_lr']
        weight_decay = training_config['weight_decay']
        beta1, beta2 = training_config['beta1'], training_config['beta2']
        
        optimizer = model.configure_optimizers(
            weight_decay, max_lr, (beta1, beta2), device_type="cuda"
        )
        
        # Initialize wandb
        if config.get('use_wandb', True):
            wandb.init(
                project=config['logging']['project'],
                name=f"{model_name}_{model.get_num_params()/1e6:.1f}M_lr{max_lr}",
                config={
                    'model': model_config,
                    'training': training_config,
                    'dataset': config['dataset'],
                    'parameters': model.get_num_params()
                }
            )
            wandb.watch(model, log='all')
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Update learning rate
            lr = get_cosine_lr(epoch, num_epochs, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Training phase
            model.train()
            total_train_loss = 0
            for batch in train_dataloader:
                (atom_seq, amino_seq, counts), coords = batch
                atom_seq = atom_seq.to(model_cfg.device)
                amino_seq = amino_seq.to(model_cfg.device)
                counts = counts.to(model_cfg.device)
                coords = coords.to(model_cfg.device)
                
                optimizer.zero_grad()
                output = model(atom_seq, amino_seq, counts)
                loss = criterion(output, coords)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item() * atom_seq.size(0)
            
            train_loss = total_train_loss / len(train_dataset)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    (atom_seq, amino_seq, counts), coords = batch
                    atom_seq = atom_seq.to(model_cfg.device)
                    amino_seq = amino_seq.to(model_cfg.device)
                    counts = counts.to(model_cfg.device)
                    coords = coords.to(model_cfg.device)
                    
                    output = model(atom_seq, amino_seq, counts)
                    loss = criterion(output, coords)
                    total_val_loss += loss.item() * atom_seq.size(0)
            
            val_loss = total_val_loss / len(val_dataset)
            val_losses.append(val_loss)
            
            # Logging
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": lr
            }
            
            if config.get('use_wandb', True):
                wandb.log(log_dict)
            
            # Print progress
            if epoch % config['logging']['log_interval'] == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:4d}/{num_epochs}: "
                      f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={lr:.2e}")
            
            # Save checkpoint
            if (config['logging'].get('save_checkpoints', False) and 
                epoch % config['logging'].get('checkpoint_interval', 100) == 0):
                checkpoint_dir = Path(config['data_paths']['results_dir']) / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), 
                          checkpoint_dir / f"{model_name}_epoch_{epoch}.pt")
        
        # Final results
        final_train_loss = np.mean(train_losses[-10:]) if len(train_losses) >= 10 else train_losses[-1]
        final_val_loss = np.mean(val_losses[-10:]) if len(val_losses) >= 10 else val_losses[-1]
        
        results[model_name] = {
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'parameters': model.get_num_params(),
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        if config.get('use_wandb', True):
            wandb.log({
                "final_train_loss": final_train_loss,
                "final_val_loss": final_val_loss
            })
            wandb.finish()
        
        print(f"\n{model_name.upper()} Results:")
        print(f"Final train loss (mean last 10 epochs): {final_train_loss:.6f}")
        print(f"Final val loss (mean last 10 epochs): {final_val_loss:.6f}")
        print(f"Parameters: {model.get_num_params()/1e6:.2f}M")
        
        # Save final model
        if config['logging'].get('save_checkpoints', False):
            checkpoint_dir = Path(config['data_paths']['results_dir']) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / f"{model_name}_final.pt")
    
    # Print comparison
    print(f"\n{'='*50}")
    print("EXPERIMENT SUMMARY - Protein Structure Prediction")
    print(f"{'='*50}")
    for model_name, result in results.items():
        print(f"{model_name.upper()}: val_loss={result['final_val_loss']:.6f}, "
              f"{result['parameters']/1e6:.2f}M params")
    
    # Analyze performance difference
    if len(results) == 2:
        models = list(results.keys())
        ht_result = results['protein_ht'] if 'protein_ht' in results else results[models[0]]
        rt_result = results['protein_rt'] if 'protein_rt' in results else results[models[1]]
        
        val_loss_improvement = rt_result['final_val_loss'] - ht_result['final_val_loss']
        param_ratio = ht_result['parameters'] / rt_result['parameters']
        
        print(f"\nHierarchical vs Regular Analysis:")
        print(f"Validation loss improvement (RT - HT): {val_loss_improvement:.6f}")
        print(f"Parameter ratio (HT/RT): {param_ratio:.2f}x")
        if val_loss_improvement > 0:
            print("✓ Hierarchical Transformer performs better (lower loss)")
        else:
            print("✗ Regular Transformer performs better (lower loss)")
    
    # Save results
    results_dir = Path(config['data_paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(results_dir / "protein_results.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for model_name, result in results.items():
            json_results[model_name] = {
                'final_train_loss': float(result['final_train_loss']),
                'final_val_loss': float(result['final_val_loss']),
                'parameters': int(result['parameters'])
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_dir / 'protein_results.json'}")
    
    return results


if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "configs" / "protein_ht_vs_rt.yaml"
    results = run_experiment(config_path)
