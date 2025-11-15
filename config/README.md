# Configuration Files

This directory contains YAML configuration files for different experimental setups.

## Available Configurations

### `local.yaml` (Default)
Balanced configuration for general use.
- Architecture: [64, 32, 16]
- Learning rate: 0.001
- Training time: ~30 seconds
- Expected R²: ~0.61-0.64

### `small_net.yaml`
Fast training with fewer parameters.
- Architecture: [32, 16]
- Batch size: 64
- Training time: ~15 seconds
- Use for: Quick iteration, testing

### `deep_net.yaml`
Maximum capacity configuration.
- Architecture: [128, 64, 32, 16]
- Higher regularization
- Training time: ~60 seconds
- Use for: Best performance

### `high_lr.yaml`
Aggressive optimization experiment.
- Learning rate: 0.01 (10x higher)
- Use for: Testing convergence speed

## Creating Custom Configs

Copy and modify any existing config:

```bash
cp local.yaml my_experiment.yaml
# Edit my_experiment.yaml
python ../src/main.py --config my_experiment.yaml
```

## Configuration Schema

```yaml
data:
  path: str                    # Dataset path
  target_cols: [str]           # Target column names
  exclude_cols: [str]          # Columns to exclude
  test_size: float            # Test set proportion
  val_size: float              # Validation set proportion

model:
  hidden_dims: [int]           # Hidden layer dimensions
  dropout_rate: float          # Dropout rate (0.0-1.0)
  activation: str              # relu, tanh, elu, leaky_relu

training:
  batch_size: int              # Training batch size
  epochs: int                  # Maximum epochs
  learning_rate: float         # Learning rate
  weight_decay: float          # L2 regularization
  patience: int                # Early stopping patience
  seed: int                    # Random seed
```

For detailed experimentation guide, see [../EXPERIMENTS.md](../EXPERIMENTS.md)
