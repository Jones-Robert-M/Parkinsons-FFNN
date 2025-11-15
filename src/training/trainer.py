"""Training loop for PyTorch models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import numpy as np
from tqdm import tqdm
import os


class Trainer:
    """
    Trainer class for PyTorch models.

    Args:
        model: PyTorch model to train
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on ('cpu' or 'cuda')
        scheduler: Optional learning rate scheduler
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": []
        }

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping: Optional['EarlyStopping'] = None,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train the model.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            early_stopping: Optional early stopping callback
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        pbar = tqdm(range(epochs), desc="Training") if verbose else range(epochs)

        for epoch in pbar:
            # Training
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # Validation
            val_loss = self.validate(val_loader)
            self.history["val_loss"].append(val_loss)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["lr"].append(current_lr)

            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Progress bar update
            if verbose:
                pbar.set_postfix({
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "lr": f"{current_lr:.6f}"
                })

            # Early stopping check
            if early_stopping is not None:
                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:
                    if verbose:
                        print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break

        return self.history

    @torch.no_grad()
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        Generate predictions.

        Args:
            data_loader: DataLoader for input data

        Returns:
            Predictions as numpy array
        """
        self.model.eval()
        predictions = []

        for X_batch, _ in data_loader:
            X_batch = X_batch.to(self.device)
            outputs = self.model(X_batch)
            predictions.append(outputs.cpu().numpy())

        return np.vstack(predictions)

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {path}")


class EarlyStopping:
    """
    Early stopping callback to stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for metrics like accuracy
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score: float, model: nn.Module):
        """
        Check if training should stop.

        Args:
            score: Current validation score
            model: Model to save if score improves
        """
        if self.mode == 'min':
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0

    def load_best_model(self, model: nn.Module):
        """Load the best model state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
