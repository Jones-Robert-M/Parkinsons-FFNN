# Experiment Guide - Model Configuration and Tuning

This document provides a comprehensive guide to configuring and experimenting with different model architectures and hyperparameters.

## Quick Reference

### Command Structure

```bash
python main.py [--config FILE] [OPTIONS]
```

### Most Common Experiments

```bash
# 1. Quick architecture test
python main.py --hidden-dims 32 16 --epochs 50 --experiment-name test

# 2. Learning rate search
python main.py --lr 0.0005 --experiment-name lr_search

# 3. Regularization tuning
python main.py --dropout 0.3 --weight-decay 0.0001 --experiment-name reg_tuned

# 4. Deep network with more capacity
python main.py --config ../config/deep_net.yaml --experiment-name deep_v1
```

## Configuration Parameters

### Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data-path` | str | `../data/raw/parkinsons_updrs.data` | Path to dataset CSV file |
| `--test-size` | float | 0.2 | Test set proportion (0.0-1.0) |
| `--val-size` | float | 0.1 | Validation set proportion (0.0-1.0) |

### Model Architecture

| Parameter | Type | Default | Description | Typical Range |
|-----------|------|---------|-------------|---------------|
| `--hidden-dims` | int[] | `[64, 32, 16]` | Hidden layer dimensions | 16-256 per layer |
| `--dropout` | float | 0.2 | Dropout rate for regularization | 0.0-0.5 |
| `--activation` | str | `relu` | Activation function | relu, tanh, elu, leaky_relu |

**Architecture Guidelines:**
- **Shallow (1-2 layers)**: Fast training, less prone to overfitting, may underfit complex patterns
- **Medium (3-4 layers)**: Good balance, works well for most datasets
- **Deep (5+ layers)**: More capacity, risk of overfitting on small datasets, needs more regularization

### Training Hyperparameters

| Parameter | Type | Default | Description | Typical Range |
|-----------|------|---------|-------------|---------------|
| `--batch-size` | int | 32 | Training batch size | 16-128 |
| `--epochs` | int | 150 | Maximum training epochs | 50-500 |
| `--lr, --learning-rate` | float | 0.001 | Learning rate | 0.0001-0.01 |
| `--weight-decay` | float | 0.00001 | L2 regularization strength | 0.0-0.001 |
| `--patience` | int | 15 | Early stopping patience (epochs) | 5-50 |
| `--seed` | int | 42 | Random seed for reproducibility | Any integer |

**Hyperparameter Guidelines:**
- **Learning Rate**: Start with 0.001, decrease if training unstable, increase if training too slow
- **Batch Size**: Larger = faster training but less noise, smaller = more regularization
- **Weight Decay**: Increase if overfitting, decrease if underfitting

### Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output-dir` | str | `../models` | Directory to save outputs |
| `--experiment-name` | str | None | Experiment name (appended to files) |
| `--no-plots` | flag | False | Disable plot generation |

## Pre-configured Experiment Templates

### 1. Small Network (`config/small_net.yaml`)

**Use Case:** Fast iteration, baseline performance

```yaml
model:
  hidden_dims: [32, 16]
  dropout_rate: 0.1

training:
  batch_size: 64
  epochs: 100
  learning_rate: 0.001
```

**Run:**
```bash
python main.py --config ../config/small_net.yaml --experiment-name small
```

**Expected:** Faster training (~15s), slightly lower accuracy, good for testing

### 2. Deep Network (`config/deep_net.yaml`)

**Use Case:** Maximum performance, more capacity

```yaml
model:
  hidden_dims: [128, 64, 32, 16]
  dropout_rate: 0.3

training:
  batch_size: 32
  epochs: 200
  learning_rate: 0.0005
  weight_decay: 0.0001
```

**Run:**
```bash
python main.py --config ../config/deep_net.yaml --experiment-name deep
```

**Expected:** Longer training (~60s), potentially better R² score

### 3. High Learning Rate (`config/high_lr.yaml`)

**Use Case:** Testing optimization aggressiveness

```yaml
training:
  learning_rate: 0.01  # 10x higher
```

**Run:**
```bash
python main.py --config ../config/high_lr.yaml --experiment-name high_lr
```

**Expected:** Faster convergence or instability

## Common Experiment Workflows

### 1. Learning Rate Grid Search

```bash
# Create experiment directory
mkdir -p ../experiments/lr_search

# Test multiple learning rates
for lr in 0.0001 0.0005 0.001 0.005 0.01; do
  python main.py \
    --lr $lr \
    --experiment-name lr_$lr \
    --output-dir ../experiments/lr_search \
    --epochs 100
done

# Compare results
ls -lh ../experiments/lr_search/
```

### 2. Architecture Exploration

```bash
mkdir -p ../experiments/architecture

# Test different depths
python main.py --hidden-dims 64 --experiment-name arch_1layer --output-dir ../experiments/architecture
python main.py --hidden-dims 64 32 --experiment-name arch_2layer --output-dir ../experiments/architecture
python main.py --hidden-dims 64 32 16 --experiment-name arch_3layer --output-dir ../experiments/architecture
python main.py --hidden-dims 128 64 32 16 --experiment-name arch_4layer --output-dir ../experiments/architecture

# Test different widths
python main.py --hidden-dims 32 16 8 --experiment-name arch_narrow --output-dir ../experiments/architecture
python main.py --hidden-dims 128 64 32 --experiment-name arch_wide --output-dir ../experiments/architecture
python main.py --hidden-dims 256 128 64 --experiment-name arch_verywide --output-dir ../experiments/architecture
```

### 3. Regularization Study

```bash
mkdir -p ../experiments/regularization

# Vary dropout
for dropout in 0.0 0.1 0.2 0.3 0.4 0.5; do
  python main.py \
    --dropout $dropout \
    --experiment-name dropout_$dropout \
    --output-dir ../experiments/regularization
done

# Vary weight decay
for wd in 0.0 0.00001 0.0001 0.001; do
  python main.py \
    --weight-decay $wd \
    --experiment-name wd_$wd \
    --output-dir ../experiments/regularization
done
```

### 4. Activation Function Comparison

```bash
mkdir -p ../experiments/activation

for act in relu tanh elu leaky_relu; do
  python main.py \
    --activation $act \
    --experiment-name act_$act \
    --output-dir ../experiments/activation
done
```

### 5. Batch Size Ablation

```bash
mkdir -p ../experiments/batch_size

for bs in 16 32 64 128; do
  python main.py \
    --batch-size $bs \
    --experiment-name bs_$bs \
    --output-dir ../experiments/batch_size
done
```

## Experiment Tracking

### Saved Artifacts Per Experiment

Each training run saves:

```
output_dir/
├── best_model_<name>.pt          # Model checkpoint
├── scaler_<name>.pkl              # Feature scaler
├── training_history_<name>.png    # Loss curves
└── config_<name>.yaml             # Exact config used
```

### Comparing Experiments

```bash
# List all experiments
ls -lh ../experiments/*/

# Check configuration differences
cat ../experiments/lr_search/config_lr_0.001.yaml
cat ../experiments/lr_search/config_lr_0.01.yaml

# View training curves
open ../experiments/lr_search/training_history_*.png
```

## Best Practices

### 1. Start Simple

```bash
# Begin with baseline
python main.py --experiment-name baseline

# Make small changes
python main.py --lr 0.0005 --experiment-name baseline_lowlr
```

### 2. Isolate Variables

Change ONE parameter at a time:

```bash
# Good: One variable
python main.py --dropout 0.3 --experiment-name test_dropout

# Bad: Multiple variables (can't isolate effect)
python main.py --dropout 0.3 --lr 0.0001 --hidden-dims 128 64 --experiment-name test_many
```

### 3. Use Reproducible Seeds

```bash
# Set same seed for fair comparison
python main.py --seed 42 --lr 0.001 --experiment-name exp1
python main.py --seed 42 --lr 0.01 --experiment-name exp2
```

### 4. Name Experiments Descriptively

```bash
# Good names
--experiment-name lr001_drop03_deep4
--experiment-name baseline_v2
--experiment-name high_reg_early_stop

# Bad names
--experiment-name test
--experiment-name exp1
--experiment-name temp
```

### 5. Track Results

Create a results log:

```bash
echo "Experiment,R2_Motor,R2_Total,RMSE_Motor,RMSE_Total" > results.csv

# After each experiment, manually add results
echo "baseline,0.61,0.64,4.97,6.26" >> results.csv
echo "high_lr,0.58,0.62,5.12,6.45" >> results.csv
```

## Troubleshooting

### Training is Too Slow

```bash
# Reduce epochs
python main.py --epochs 50

# Increase batch size
python main.py --batch-size 128

# Simplify architecture
python main.py --hidden-dims 32 16
```

### Model is Overfitting

```bash
# Increase dropout
python main.py --dropout 0.4

# Increase weight decay
python main.py --weight-decay 0.001

# Simplify model
python main.py --hidden-dims 32 16

# Increase patience for better generalization
python main.py --patience 25
```

### Model is Underfitting

```bash
# Increase capacity
python main.py --hidden-dims 128 64 32

# Reduce dropout
python main.py --dropout 0.1

# Reduce weight decay
python main.py --weight-decay 0.00001

# Train longer
python main.py --epochs 300 --patience 30
```

### Training is Unstable

```bash
# Reduce learning rate
python main.py --lr 0.0001

# Add more regularization
python main.py --weight-decay 0.0001

# Use gentler activation
python main.py --activation tanh
```

## Advanced: Custom Config Files

Create your own YAML configuration:

```yaml
# config/my_experiment.yaml
data:
  path: "../data/raw/parkinsons_updrs.data"
  target_cols: ["motor_UPDRS", "total_UPDRS"]
  test_size: 0.2
  val_size: 0.1

model:
  hidden_dims: [96, 48, 24]  # Custom architecture
  dropout_rate: 0.25
  activation: "elu"           # Different activation

training:
  batch_size: 48
  epochs: 175
  learning_rate: 0.0008
  weight_decay: 0.00005
  patience: 18
  seed: 42
```

Run with:
```bash
python main.py --config ../config/my_experiment.yaml --experiment-name my_exp
```

## Summary

- **Use pre-configured templates** for quick testing
- **CLI arguments** for rapid experimentation
- **YAML configs** for reproducible experiments
- **Experiment naming** for organization
- **One variable at a time** for interpretability
- **Track results** systematically

Happy experimenting! 🚀
