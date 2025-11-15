"""Main script for training and evaluating the feedforward neural network."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import load_data, preprocess_data, split_data, ParkinsonsDataset
from models import FeedForwardNN
from training import Trainer, compute_metrics, EarlyStopping
from training.metrics import print_metrics


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_training_history(history: dict, save_path: str = None):
    """
    Plot training and validation loss.

    Args:
        history: Training history dictionary
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(epochs, history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Learning rate plot
    ax2.plot(epochs, history['lr'], color='green', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to {save_path}")
    else:
        plt.show()


def main(config_path: str = "../config/local.yaml"):
    """
    Main training pipeline.

    Args:
        config_path: Path to configuration YAML file
    """
    # Load configuration
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        print(f"Loaded config from {config_path}")
    else:
        print(f"Config file not found at {config_path}, using defaults")
        config = {}

    # Extract config values with defaults
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})

    # Data parameters
    data_path = data_config.get('path', '../data/raw/parkinsons_updrs.data')
    target_cols = data_config.get('target_cols', ['motor_UPDRS', 'total_UPDRS'])
    test_size = data_config.get('test_size', 0.2)
    val_size = data_config.get('val_size', 0.1)

    # Model parameters
    hidden_dims = model_config.get('hidden_dims', [64, 32, 16])
    dropout_rate = model_config.get('dropout_rate', 0.2)
    activation = model_config.get('activation', 'relu')

    # Training parameters
    batch_size = training_config.get('batch_size', 32)
    epochs = training_config.get('epochs', 100)
    learning_rate = training_config.get('learning_rate', 0.001)
    weight_decay = training_config.get('weight_decay', 1e-5)
    patience = training_config.get('patience', 15)
    seed = training_config.get('seed', 42)

    # Set random seed
    set_seed(seed)

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # -------------------------------------------------------------------------
    # 1. Load and preprocess data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Loading and Preprocessing Data")
    print("=" * 60)

    df = load_data(data_path)
    X, y, scaler = preprocess_data(df, target_cols=target_cols)
    data_splits = split_data(X, y, test_size=test_size, val_size=val_size, random_state=seed)

    # Create datasets
    train_dataset = ParkinsonsDataset(data_splits['X_train'], data_splits['y_train'], device)
    val_dataset = ParkinsonsDataset(data_splits['X_val'], data_splits['y_val'], device)
    test_dataset = ParkinsonsDataset(data_splits['X_test'], data_splits['y_test'], device)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------------------------------------------------------
    # 2. Build model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Building Model")
    print("=" * 60)

    model = FeedForwardNN(
        input_dim=train_dataset.n_features,
        hidden_dims=hidden_dims,
        output_dim=train_dataset.n_targets,
        dropout_rate=dropout_rate,
        activation=activation
    )

    print(model)
    print(f"\nModel has {model.get_num_parameters():,} trainable parameters")

    # -------------------------------------------------------------------------
    # 3. Setup training
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Training Setup")
    print("=" * 60)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0001)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )

    print(f"Criterion: {criterion.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__} (lr={learning_rate})")
    print(f"Scheduler: ReduceLROnPlateau")
    print(f"Early Stopping: patience={patience}")

    # -------------------------------------------------------------------------
    # 4. Train model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Training Model")
    print("=" * 60)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping=early_stopping,
        verbose=True
    )

    # Load best model
    early_stopping.load_best_model(model)
    print("\nLoaded best model from early stopping")

    # -------------------------------------------------------------------------
    # 5. Evaluate model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Evaluation")
    print("=" * 60)

    # Predictions
    y_train_pred = trainer.predict(train_loader)
    y_val_pred = trainer.predict(val_loader)
    y_test_pred = trainer.predict(test_loader)

    # Compute metrics
    train_metrics = compute_metrics(data_splits['y_train'], y_train_pred)
    val_metrics = compute_metrics(data_splits['y_val'], y_val_pred)
    test_metrics = compute_metrics(data_splits['y_test'], y_test_pred)

    # Print metrics
    print_metrics(train_metrics, "Training Metrics")
    print_metrics(val_metrics, "Validation Metrics")
    print_metrics(test_metrics, "Test Metrics")

    # -------------------------------------------------------------------------
    # 6. Save results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Saving Results")
    print("=" * 60)

    # Create output directory
    output_dir = "../models"
    os.makedirs(output_dir, exist_ok=True)

    # Save model checkpoint
    checkpoint_path = os.path.join(output_dir, "best_model.pt")
    trainer.save_checkpoint(checkpoint_path)

    # Save training plot
    plot_path = os.path.join(output_dir, "training_history.png")
    plot_training_history(history, save_path=plot_path)

    # Save scaler
    import joblib
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train feedforward neural network")
    parser.add_argument(
        '--config',
        type=str,
        default='../config/local.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()

    main(config_path=args.config)
