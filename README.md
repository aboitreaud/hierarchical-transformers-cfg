# Hierarchical Transformers for Context-Free Grammars

This repository contains reproducible experiments comparing Regular Transformers (RT) and Hierarchical Transformers (HT) on Context-Free Grammar (CFG) tasks and protein structure prediction.

## Quick Start

### Option 1: Run All Experiments
```bash
# Clone and setup
git clone https://github.com/aboitreaud/hierarchical-transformers-cfg.git

# Or clone via SSH
# git clone git@github.com:aboitreaud/hierarchical-transformers-cfg.git
cd hierarchical-transformers-cfg

# Install dependencies
pip install -r requirements.txt

# Run all experiments (this will take several hours)
./scripts/reproduce_all.sh
```

### Option 2: Run Individual Experiments
```bash
# HT-2L vs RT-4L comparison
python experiments/scripts/run_ht_2l_vs_rt_4l.py

# HT-4L vs RT-8L parameter-matched comparison  
python experiments/scripts/run_ht_4l_vs_rt_8l.py

# Depth sweep analysis (CFG-3L to CFG-11L)
python experiments/scripts/run_depth_sweep.py

# Binary HT vs RT on deep CFG-9L
python experiments/scripts/run_ht_binary_vs_rt_cfg9l.py

# Protein structure prediction (requires dataset)
python experiments/scripts/run_protein_ht_vs_rt.py
```

## Experiments Overview

The `scripts/reproduce_all.sh` script runs the following experiments:

### 1. RT and HT on CFG-7L
This experiment compares RT and HT performance on 7-level CFG. We use cosine learning rate schedule from 6e-4 to 6e-5, training for 100 epochs with 5k sentences per epoch. The models are 6-layer RT vs 4-transformer HT.

Script: `run_ht_2l_vs_rt_4l.py`

### 2. HT-4L vs RT-8L
This is parameter-matched comparison of hierarchical vs depth. We compare 4-layer HT vs 8-layer RT with similar parameter counts to see which architecture is more efficient.

Script: `run_ht_4l_vs_rt_8l.py`

### 3. Depth Sweep Analysis
This experiment studies transformer accuracy vs CFG depth. We test CFG depths from 3L through 11L and analyze both transformer performance and spectral clustering heuristics for reproduction.

Script: `run_depth_sweep.py`

### 4. HT-Binary vs RT on CFG-9L
This compares binary hierarchy vs flat transformer on deep grammar. We use 2-level HT vs standard RT on 9-level CFG to test performance on very deep hierarchical structures.

Script: `run_ht_binary_vs_rt_cfg9l.py`

### 5. Protein HT vs RT with Kabsch-MSE
This experiment compares protein structure prediction using hierarchical (atom+residue) vs regular transformer. We use Kabsch-aligned MSE loss for 3D coordinate prediction.

Script: `run_protein_ht_vs_rt.py`

## Project Structure

```
hierarchical-transformers-cfg/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   ├── cfg.py               # Context-Free Grammar implementation
│   │   ├── transformers.py      # RT and HT models
│   │   └── protein_models.py    # Protein-specific models
│   └── utils/                   # Training utilities
│       └── training.py          # Training loops and helpers
├── experiments/                 # Experiment configurations and scripts
│   ├── configs/                # YAML configuration files
│   ├── scripts/                # Individual experiment scripts
│   └── results/                # Experiment outputs (created during runs)
├── scripts/                    # Main execution scripts
│   ├── reproduce_all.sh        # Master script to run all experiments
│   └── setup_environment.sh    # Environment setup script
├── requirements.txt            # Python dependencies
└── README.md
```

## Installation and Setup

### Requirements
- Python 3.8 or later
- CUDA-capable GPU (recommended for faster training) with torch 2.0.0 or later
- At least 16GB RAM for larger experiments

### Installation Steps
```bash
# Create virtual environment (recommended)
python -m venv cfg-env
source cfg-env/bin/activate  # On Windows: cfg-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or use the setup script (recommended)
./scripts/setup_environment.sh

```

### Weights and Biases Setup (Optional)
If you want to track experiments with Weights and Biases:
```bash
wandb login
```

## Model Architectures

### Regular Transformer (RT)
This is standard GPT-style transformer with token embeddings and positional embeddings. It uses multi-head self-attention with causal masking and feed-forward networks with GELU activation.

### Hierarchical Transformer (HT)
This model has multiple transformer levels that correspond to CFG hierarchy. It uses parent embedders for level transitions and dimension reducers for multi-level fusion. The architecture is specialized for hierarchical sequence modeling.

### Protein Models
- **Protein HT**: Uses separate transformers for atoms and residues
- **Protein RT**: Uses single transformer on atom sequences only
- **Kabsch-MSE Loss**: Rotation and translation invariant loss function

## Configuration

Each experiment uses YAML configuration files in `experiments/configs/`:

```yaml
# Example: ht_2l_vs_rt_4l.yaml
experiment_name: "ht_2l_vs_rt_4l"
cfg:
  L: 7  # CFG depth
  ns: [1, 3, 3, 3, 5, 5, 9, 10]  # Symbols per level
  nr: [2, 2, 2, 2, 2, 2, 2]      # Rules per symbol
  T: [2, 2, 2, 2, 2, 4, 4]       # Rule lengths

models:
  rt:
    type: "GPT"
    n_layer: 6
    n_head: 6
    head_size: 64
  
training:
  num_epochs: 100
  max_lr: 6e-4
  min_lr: 6e-5
```

## Expected Results

### CFG Tasks
- **Accuracy**: Percentage of grammatically correct generated sentences
- **Per-level Accuracy**: Correctness at each CFG level
- **Parameter Efficiency**: Accuracy per million parameters

### Protein Tasks  
- **Kabsch-MSE**: Rotation-invariant coordinate prediction error
- **Training/Validation Loss**: Learning curves over epochs

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size in config files
batch_size: 50  # Instead of 100
```

**Missing Protein Dataset**
```bash
# The protein experiment requires protein_dataset.pkl
# Skip with: python run_protein_ht_vs_rt.py --skip-if-missing
```

**Slow Training**
```bash
# Reduce epochs for testing
num_epochs: 10  # Instead of 100
```

### Performance Tips
- Use CUDA if available (gives 10-100x speedup)
- Monitor GPU memory usage during training
- Use mixed precision training for larger models
- Adjust batch sizes based on available memory

## Citation

If you use this code in your research, please cite:

```bibtex
@article{cfg-transformers-2024,
  title={Hierarchical vs Regular Transformers on Context-Free Grammar Tasks},
  author={[Author Names]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original CFG implementation and experimental design
- PyTorch team for the deep learning framework
- Weights and Biases for experiment tracking
- Research community for hierarchical transformer architectures

## Support

For questions or issues:
1. Check the [Issues](../../issues) page
2. Review the troubleshooting section above
3. Create a new issue with detailed information
