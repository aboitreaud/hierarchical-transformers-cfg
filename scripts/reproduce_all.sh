#!/bin/bash

# CFG Project - Reproduce All Experiments
# This script runs all the experiments described in the thesis

set -e  # Exit on any error

echo "=========================================="
echo "CFG Project - Reproducing All Experiments"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "scripts/reproduce_all.sh" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
python -c "import torch, wandb, yaml, numpy, jaxtyping" 2>/dev/null || {
    echo "Error: Missing dependencies. Please run: pip install -r requirements.txt"
    exit 1
}

# Check CUDA availability
if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep -q "True"; then
    echo "✓ CUDA is available"
else
    echo "⚠ CUDA not available - experiments will run on CPU (much slower)"
fi

# Create results directories
mkdir -p experiments/results/{ht_2l_vs_rt_4l,ht_4l_vs_rt_8l,depth_sweep,ht_binary_vs_rt_cfg9l,protein_ht_vs_rt}

echo ""
echo "Starting experiments..."
echo "Note: Each experiment will log to Weights & Biases if configured"
echo ""

# Experiment 1: RT and HT on CFG-7L
echo "=========================================="
echo "Experiment 1: RT and HT on CFG-7L"
echo "Cosine LR 6e-4→6e-5, 100 epochs, 5k sentences/epoch"
echo "=========================================="
cd experiments/scripts
python run_ht_2l_vs_rt_4l.py
cd ../..
echo "✓ HT-2L vs RT-4L experiment completed"
echo ""

# Experiment 2: HT-4L vs RT-8L
echo "=========================================="
echo "Experiment 2: HT-4L vs RT-8L Comparison"
echo "Parameter-matched comparison"
echo "=========================================="
cd experiments/scripts
python run_ht_4l_vs_rt_8l.py
cd ../..
echo "✓ HT-4L vs RT-8L experiment completed"
echo ""

# Experiment 3: Depth Sweep
echo "=========================================="
echo "Experiment 3: Depth Sweep Analysis"
echo "Transformer Accuracy vs CFG Depth (CFG-3L to CFG-11L)"
echo "=========================================="
cd experiments/scripts
python run_depth_sweep.py
cd ../..
echo "✓ Depth sweep experiment completed"
echo ""

# Experiment 4: HT-Binary vs RT on CFG-9L
echo "=========================================="
echo "Experiment 4: HT-Binary vs RT on CFG-9L"
echo "Binary hierarchy vs flat transformer on deep grammar"
echo "=========================================="
cd experiments/scripts
python run_ht_binary_vs_rt_cfg9l.py
cd ../..
echo "✓ HT-Binary vs RT CFG-9L experiment completed"
echo ""

# Experiment 5: Protein HT vs RT (if dataset available)
echo "=========================================="
echo "Experiment 5: Protein HT vs RT with Kabsch-MSE"
echo "Protein structure prediction comparison"
echo "=========================================="

if [ -f "protein_dataset.pkl" ]; then
    echo "Using existing protein dataset..."
    cd experiments/scripts
    python run_protein_ht_vs_rt.py
    cd ../..
    echo "✓ Protein HT vs RT experiment completed"
else
    echo "Protein dataset not found. Creating from atom3d..."
    python scripts/prepare_protein_dataset.py
    if [ -f "protein_dataset.pkl" ]; then
        cd experiments/scripts
        python run_protein_ht_vs_rt.py
        cd ../..
        echo "✓ Protein HT vs RT experiment completed"
    else
        echo "⚠ Failed to create protein dataset"
        echo "  To manually prepare dataset, run:"
        echo "    python scripts/prepare_protein_dataset.py"
    fi
fi
echo ""

# Summary
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "=========================================="
echo ""
echo "Results have been saved to:"
echo "  - experiments/results/ht_2l_vs_rt_4l/"
echo "  - experiments/results/ht_4l_vs_rt_8l/"
echo "  - experiments/results/depth_sweep/"
echo "  - experiments/results/ht_binary_vs_rt_cfg9l/"
echo "  - experiments/results/protein_ht_vs_rt/ (if run)"
echo ""
echo "If using Weights & Biases, check your dashboard for detailed logs and metrics."
echo ""
echo "To run individual experiments:"
echo "  python experiments/scripts/run_ht_2l_vs_rt_4l.py"
echo "  python experiments/scripts/run_ht_4l_vs_rt_8l.py"
echo "  python experiments/scripts/run_depth_sweep.py"
echo "  python experiments/scripts/run_ht_binary_vs_rt_cfg9l.py"
echo "  python experiments/scripts/run_protein_ht_vs_rt.py"
echo ""
echo "For more information, see README.md"
