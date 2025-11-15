"""Data preprocessing utilities for Parkinson's dataset."""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional


def load_data(data_path: str = "../data/raw/parkinsons_updrs.data") -> pd.DataFrame:
    """
    Load Parkinson's telemonitoring dataset.

    Args:
        data_path: Path to the CSV data file

    Returns:
        DataFrame containing the dataset
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


def preprocess_data(
    df: pd.DataFrame,
    target_cols: list = ["motor_UPDRS", "total_UPDRS"],
    exclude_cols: list = ["subject#"],
    scale_features: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    Preprocess the Parkinson's dataset.

    Args:
        df: Input DataFrame
        target_cols: List of target column names for regression
        exclude_cols: Columns to exclude from features
        scale_features: Whether to standardize features

    Returns:
        Tuple of (features, targets, scaler)
    """
    # Separate features and targets
    feature_cols = [col for col in df.columns
                   if col not in target_cols and col not in exclude_cols]

    X = df[feature_cols].values
    y = df[target_cols].values

    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {feature_cols}")

    # Check for missing values
    if np.isnan(X).any():
        print("Warning: Found NaN values in features. Filling with median.")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)

    # Scale features
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print("Features standardized (mean=0, std=1)")

    return X, y, scaler


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Split data into train, validation, and test sets.

    Args:
        X: Feature array
        y: Target array
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing train/val/test splits
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: separate validation from training
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
