#!/bin/bash

# CFG Project - Environment Setup Script
# This script sets up the environment for running experiments

set -e

echo "=========================================="
echo "CFG Project - Environment Setup"
echo "=========================================="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "Error: Python 3.8 or higher is required"
    exit 1
fi

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠ No virtual environment detected. Consider creating one:"
    echo "  python -m venv cfg-env"
    echo "  source cfg-env/bin/activate"
    echo ""
fi

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('Running on CPU - experiments will be slower')
"

# Check Weights & Biases
echo ""
echo "Checking Weights & Biases setup..."
if wandb --version > /dev/null 2>&1; then
    echo "✓ Weights & Biases installed"
    if [ -f ~/.netrc ] && grep -q "machine api.wandb.ai" ~/.netrc; then
        echo "✓ W&B appears to be configured"
    else
        echo "⚠ W&B not configured. Run 'wandb login' to set up experiment tracking"
    fi
else
    echo "✗ Weights & Biases not found"
fi

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p experiments/results/{ht_2l_vs_rt_4l,ht_4l_vs_rt_8l,depth_sweep,ht_binary_vs_rt_cfg9l,protein_ht_vs_rt}
mkdir -p experiments/results/{ht_2l_vs_rt_4l,ht_4l_vs_rt_8l,depth_sweep,ht_binary_vs_rt_cfg9l,protein_ht_vs_rt}/checkpoints

# Make scripts executable
chmod +x scripts/*.sh

echo ""
echo "=========================================="
echo "Environment Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Configure W&B (optional): wandb login"
echo "2. Run all experiments: ./scripts/reproduce_all.sh"
echo "3. Or run individual experiments:"
echo "   python experiments/scripts/run_ht_2l_vs_rt_4l.py"
echo ""
echo "For more information, see README.md"
