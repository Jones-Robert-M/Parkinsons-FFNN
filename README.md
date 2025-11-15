# Parkinson's Disease Prediction - Feedforward Neural Network

A complete PyTorch-based machine learning pipeline for predicting Parkinson's disease progression metrics (motor_UPDRS and total_UPDRS) from voice telemonitoring data. This project demonstrates end-to-end ML workflow including exploratory data analysis, model training, evaluation, and prediction analysis.

## 🎯 Project Overview

This project implements a shallow feedforward neural network for regression tasks on the Parkinson's Telemonitoring dataset. The goal is to predict disease progression scores from voice measurements, providing a non-invasive method for monitoring Parkinson's disease.

### Key Features

- ✅ **Complete ML Pipeline**: Data preprocessing → Model training → Evaluation → Prediction
- ✅ **Comprehensive EDA**: Statistical analysis, PCA, correlation studies, outlier detection
- ✅ **Production-Ready**: Modular code structure with configuration management
- ✅ **Reproducible**: Fixed random seeds, versioned dependencies, documented experiments
- ✅ **Visualization**: Training curves, residual analysis, prediction quality assessment
- ✅ **Best Practices**: Early stopping, learning rate scheduling, proper train/val/test splits

## 📁 Project Structure

```
Parkinsons-FFNN/
├── config/
│   ├── local.yaml              # Local training configuration
│   └── prod.yaml               # Production configuration
├── data/
│   ├── raw/                    # Raw dataset files
│   └── predictions/            # Model predictions output
├── models/                     # Saved models and artifacts
│   ├── best_model.pt          # Trained model checkpoint
│   ├── scaler.pkl             # Feature scaler
│   └── training_history.png   # Training curves visualization
├── notebooks/
│   ├── eda.ipynb              # Exploratory data analysis (12 sections)
│   └── prediction_analysis.ipynb  # Model prediction analysis (10 sections)
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py    # Data loading and preprocessing
│   │   └── dataset.py          # PyTorch Dataset class
│   ├── models/
│   │   ├── __init__.py
│   │   └── ffnn.py            # Feedforward neural network
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training loop with early stopping
│   │   └── metrics.py         # Evaluation metrics
│   ├── pipelines/
│   │   └── __init__.py
│   ├── main.py                # Main training script
│   ├── predict.py             # Inference script
│   └── utils.py               # Helper functions
├── tests/
│   └── __init__.py            # Unit tests
├── requirements.txt           # Project dependencies
├── CLAUDE.md                  # Project workflow documentation
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

Explore the dataset with comprehensive statistical analysis:
```bash
jupyter lab notebooks/eda.ipynb
```

**EDA Notebook Contents:**
- Dataset overview and statistics
- Target variable analysis
- Feature distributions
- Correlation analysis
- Principal Component Analysis (PCA)
- PCA component loadings
- 2D/3D PCA visualizations
- Outlier detection (Isolation Forest)
- Pairwise feature relationships
- Time-series analysis
- Demographic analysis
- Key findings and recommendations

### 4. Prediction Analysis

Analyze model predictions and performance:
```bash
jupyter lab notebooks/prediction_analysis.ipynb
```

**Prediction Analysis Contents:**
- Overall performance metrics
- Prediction vs actual scatter plots
- Residual analysis
- Error distribution by prediction range
- Q-Q plots for normality checks
- Per-subject prediction accuracy
- Temporal prediction tracking
- Error correlation analysis
- Feature impact on errors
- Comprehensive summary

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

## 📊 Evaluation Metrics

The model is evaluated using:
- **MSE** (Mean Squared Error) - Average squared difference between predictions and actuals
- **RMSE** (Root Mean Squared Error) - Square root of MSE, in original units
- **MAE** (Mean Absolute Error) - Average absolute difference
- **R²** (Coefficient of Determination) - Proportion of variance explained (0-1 scale)
- **MAPE** (Mean Absolute Percentage Error) - Percentage-based error metric

## 🚀 Expected Results

Based on the trained model:

**Test Set Performance:**
- Motor UPDRS: R² ≈ 0.61, RMSE ≈ 4.97, MAE ≈ 3.85
- Total UPDRS: R² ≈ 0.64, RMSE ≈ 6.26, MAE ≈ 4.75

**Training Details:**
- Epochs: ~78 (early stopping)
- Learning rate: 0.001 → 0.000125 (adaptive)
- Parameters: 4,146 trainable weights
- Training time: ~30 seconds on CPU

## 🔬 Experimental Workflow

1. **Data Exploration** (`notebooks/eda.ipynb`)
   - Load and analyze the Parkinson's dataset
   - Perform PCA and correlation studies
   - Identify outliers and patterns

2. **Model Training** (`src/main.py`)
   - Preprocess and split data (70/10/20)
   - Train FFNN with early stopping
   - Save best model and artifacts

3. **Prediction Generation** (`src/predict.py`)
   - Load trained model and scaler
   - Generate predictions on dataset
   - Save to `data/predictions/`

4. **Results Analysis** (`notebooks/prediction_analysis.ipynb`)
   - Evaluate prediction quality
   - Analyze residuals and errors
   - Generate insights and recommendations

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Modifying Architecture
Edit `config/local.yaml` to experiment with:
- Hidden layer dimensions
- Dropout rates
- Activation functions
- Learning rates
- Batch sizes

### Adding Features
The modular structure allows easy extension:
- New models in `src/models/`
- Custom metrics in `src/training/metrics.py`
- Additional preprocessing in `src/data/preprocessing.py`

## 📝 License

See LICENSE file for details.

## 🙏 Acknowledgments

This project demonstrates best practices for:
- Reproducible ML research
- Clean code architecture
- Comprehensive evaluation
- Transparent reporting

## 📚 Citation

**Dataset Source:**
```
A Tsanas, MA Little, PE McSharry, LO Ramig (2009)
'Accurate telemonitoring of Parkinson's disease progression by non-invasive speech tests',
IEEE Transactions on Biomedical Engineering
```

**Repository:**
```
Parkinson's FFNN Project
https://github.com/your-username/Parkinsons-FFNN
```

---

## 📖 Quick Start Guide

```bash
# 1. Setup
git clone <repository-url>
cd Parkinsons-FFNN
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Explore data
jupyter lab notebooks/eda.ipynb

# 3. Train model
cd src
python main.py

# 4. Generate predictions
python predict.py --data ../data/raw/parkinsons_updrs.data --output ../data/predictions/predictions.csv

# 5. Analyze results
cd ..
jupyter lab notebooks/prediction_analysis.ipynb
```

**That's it!** You now have a trained neural network with complete analysis. 🎉
