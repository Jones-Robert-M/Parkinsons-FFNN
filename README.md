# Parkinson's Disease Prediction - Feedforward Neural Network

A PyTorch-based feedforward neural network for predicting Parkinson's disease progression metrics (motor_UPDRS and total_UPDRS) from telemonitoring data.

## Project Structure

```
Landmine-FFNN/
├── config/
│   ├── local.yaml              # Local training configuration
│   └── prod.yaml               # Production configuration
├── data/
│   └── raw/                    # Raw dataset files
├── models/                     # Saved models and artifacts
├── notebooks/
│   └── eda.ipynb              # Exploratory data analysis
├── src/
│   ├── data/
│   │   ├── preprocessing.py    # Data loading and preprocessing
│   │   └── dataset.py          # PyTorch Dataset class
│   ├── models/
│   │   └── ffnn.py            # Feedforward neural network
│   ├── training/
│   │   ├── trainer.py         # Training loop
│   │   └── metrics.py         # Evaluation metrics
│   ├── main.py                # Main training script
│   └── predict.py             # Inference script
├── tests/                     # Unit tests
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## Dataset

**Parkinson's Telemonitoring Dataset**
- **Instances:** 5,875 voice recordings
- **Features:** 22 (including age, gender, time, and various voice measurements)
- **Targets:** motor_UPDRS and total_UPDRS scores
- **Source:** UCI Machine Learning Repository

## Installation

1. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training

Train the model with default configuration:
```bash
cd src
python main.py
```

Train with custom configuration:
```bash
python main.py --config ../config/local.yaml
```

### 2. Making Predictions

Use the trained model for inference:
```bash
python predict.py --data ../data/raw/parkinsons_updrs.data --output data/prediction/predictions.csv
```

### 3. Exploratory Data Analysis

Open and run the Jupyter notebook:
```bash
jupyter lab notebooks/eda.ipynb
```

## Model Architecture

**Feedforward Neural Network:**
- Input layer: 19 features (after preprocessing)
- Hidden layers: [64, 32, 16] neurons with ReLU activation
- Batch normalization after each hidden layer
- Dropout (0.2) for regularization
- Output layer: 2 neurons (motor_UPDRS, total_UPDRS)

**Training Features:**
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam with weight decay
- Learning rate scheduler: ReduceLROnPlateau
- Early stopping with patience=15
- Train/Val/Test split: 70/10/20

## Configuration

Edit `config/local.yaml` to customize:

```yaml
data:
  target_cols: ["motor_UPDRS", "total_UPDRS"]
  test_size: 0.2
  val_size: 0.1

model:
  hidden_dims: [64, 32, 16]
  dropout_rate: 0.2
  activation: "relu"

training:
  batch_size: 32
  epochs: 150
  learning_rate: 0.001
  patience: 15
```

## Results

After training, the following artifacts are saved to `models/`:
- `best_model.pt` - Model checkpoint with best validation loss
- `scaler.pkl` - StandardScaler for feature normalization
- `training_history.png` - Training/validation loss curves

## Evaluation Metrics

The model is evaluated using:
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)

## Development

Run tests:
```bash
pytest tests/
```

## License

See LICENSE file for details.

## Citation

Dataset source:
```
A Tsanas, MA Little, PE McSharry, LO Ramig (2009)
'Accurate telemonitoring of Parkinson's disease progression by non-invasive speech tests',
IEEE Transactions on Biomedical Engineering
```