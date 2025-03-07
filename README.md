# Tumor Classification CNN

A convolutional neural network (CNN) for classifying tumor images using PyTorch.

## Project Structure

- `model.py`: Contains the CNN architecture with BatchNorm and Dropout layers
- `data_utils.py`: Utilities for loading and preprocessing the tumor dataset
- `train.py`: Training script with progress bars and visualization
- `experiment.py`: Experiment class for hyperparameter tuning and model optimization

## Model Architecture

The CNN model includes:
- 4 convolutional blocks with BatchNorm and MaxPooling
- 2 fully connected layers with BatchNorm and Dropout
- Kaiming initialization for weights

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Single Model Training

Train a single model with default parameters:
```bash
python train.py
```

Optional arguments:
- `--batch_size`: Batch size for training (default: 32)
- `--num_epochs`: Number of training epochs (default: 10)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--dropout_rate`: Dropout probability (default: 0.5)

### Hyperparameter Tuning

Run experiments with different hyperparameter configurations:

```python
from experiment import Experiment

# Define hyperparameter grid
param_grid = {
    'batch_size': [16, 32],
    'learning_rate': [0.001, 0.0001],
    'optimizer': ['Adam', 'SGD'],
    'num_epochs': [10],
    'dropout_rate': [0.3, 0.5],
    'optimizer_params': [
        {},  # Default parameters for Adam
        {'momentum': 0.9}  # Parameters for SGD
    ]
}

# Create and run experiment
experiment = Experiment('tumor_classification_v1')
results = experiment.run_experiment(param_grid)

# Load best model
best_model, best_config = experiment.load_best_model()
```

## Dataset

This project uses the "youngp5/tumors" dataset from Hugging Face, which contains tumor images for binary classification.

## Experiment Results

The experiment framework generates:

### Directory Structure
```
experiments/
└── experiment_name_timestamp/
    ├── experiment_results.csv      # All configurations and their results
    ├── best_config.json           # Best performing configuration
    ├── model_0/
    │   ├── config.json            # Model configuration
    │   ├── training_history.csv   # Training metrics
    │   ├── best_model_acc_X.pth   # Best checkpoint by accuracy
    │   ├── best_model_final.pth   # Final best model
    │   └── training_history.png   # Training visualization
    └── model_N/                   # Results for each configuration
```

### Tracked Metrics
- Training and validation accuracy
- Training and validation loss
- Best model checkpoints
- Training time
- Hyperparameter configurations

### Visualization
- Training/validation accuracy curves
- Training/validation loss curves
- Saved for each model configuration

## Model Selection

The experiment framework automatically:
1. Trains models with different hyperparameter combinations
2. Tracks performance metrics
3. Saves the best performing model
4. Generates comprehensive reports
5. Provides easy access to the best configuration

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.