"""Inference script for making predictions with trained model."""

import torch
import numpy as np
import pandas as pd
import joblib
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import FeedForwardNN


def load_model_and_scaler(model_path: str, scaler_path: str, device: str = "cpu"):
    """
    Load trained model and scaler.

    Args:
        model_path: Path to model checkpoint
        scaler_path: Path to saved scaler
        device: Device to load model on

    Returns:
        Tuple of (model, scaler)
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Reconstruct model (need to know architecture)
    # In production, save model config in checkpoint
    model = FeedForwardNN(
        input_dim=19,  # From preprocessing (22 - 3 excluded columns)
        hidden_dims=[64, 32, 16],
        output_dim=2,
        dropout_rate=0.2,
        activation="relu"
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load scaler
    scaler = joblib.load(scaler_path)

    print(f"Model loaded from {model_path}")
    print(f"Scaler loaded from {scaler_path}")

    return model, scaler


def predict(
    model: torch.nn.Module,
    scaler,
    X: np.ndarray,
    device: str = "cpu"
) -> np.ndarray:
    """
    Make predictions on new data.

    Args:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        X: Feature array to predict on
        device: Device to run predictions on

    Returns:
        Predictions array
    """
    # Scale features
    X_scaled = scaler.transform(X)

    # Convert to tensor
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    # Predict
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument(
        '--model',
        type=str,
        default='../models/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--scaler',
        type=str,
        default='../models/scaler.pkl',
        help='Path to scaler file'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Path to save predictions'
    )

    args = parser.parse_args()

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model and scaler
    model, scaler = load_model_and_scaler(args.model, args.scaler, device)

    # Load input data
    df = pd.read_csv(args.data)
    print(f"\nLoaded {len(df)} samples from {args.data}")

    # Extract features (exclude target and subject columns)
    exclude_cols = ['subject#', 'motor_UPDRS', 'total_UPDRS']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].values

    # Make predictions
    predictions = predict(model, scaler, X, device)

    # Create output dataframe
    output_df = df.copy()
    output_df['pred_motor_UPDRS'] = predictions[:, 0]
    output_df['pred_total_UPDRS'] = predictions[:, 1]

    # Save predictions
    output_df.to_csv(args.output, index=False)
    print(f"\nPredictions saved to {args.output}")

    # Display sample
    print("\nSample predictions:")
    print(output_df[['motor_UPDRS', 'pred_motor_UPDRS', 'total_UPDRS', 'pred_total_UPDRS']].head())


if __name__ == "__main__":
    main()
