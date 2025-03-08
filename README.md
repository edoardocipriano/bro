# Tumor Classification CNN

A convolutional neural network (CNN) for classifying tumor images using PyTorch.

## Project Structure

- `src/model.py`: Contains the CNN architecture with BatchNorm and Dropout layers
- `src/data_utils.py`: Utilities for loading and preprocessing the tumor dataset
- `src/train.py`: Training script with progress bars and visualization
- `src/evaluate.py`: Evaluation utilities for model assessment and visualization
- `src/experiment.py`: Experiment class for hyperparameter tuning and model optimization
- `src/main.py`: Main entry point with CLI for running training and experiments

## Model Architecture

The project includes two CNN model variants:
- **Lightweight CNN**: A smaller, faster model with:
  - Initial pooling to reduce computation
  - 3 convolutional blocks with BatchNorm and MaxPooling
  - Global Average Pooling
  - Single fully connected layer with Dropout

## Dataset

This project uses the "youngp5/tumors" dataset from Hugging Face, which contains tumor images for classification.

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Single Model Training

Train a single model with default parameters:
```bash
python src/main.py
```

Optional arguments:
- `--batch_size`: Batch size for training (default: 32)
- `--num_epochs`: Number of training epochs (default: 25)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--dropout_rate`: Dropout probability (default: 0.5)
- `--model_type`: Model architecture to use (default: 'light')

### Hyperparameter Tuning

Run experiments with different hyperparameter configurations:

```python
from src.experiment import Experiment

# Define hyperparameter grid
param_grid = {
    'batch_size': [16, 32],
    'learning_rate': [0.001, 0.0001],
    'optimizer': ['Adam', 'SGD'],
    'num_epochs': [25],
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

# Evaluate best model
evaluation_results = experiment.evaluate_best_model()
```

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
- Confusion matrices
- Classification reports

### Visualization
- Training/validation accuracy curves
- Training/validation loss curves
- Confusion matrices
- Saved for each model configuration

## Model Evaluation

The project includes comprehensive evaluation tools:
- Classification metrics (precision, recall, F1-score)
- Confusion matrix visualization
- Single image prediction functionality
- Model loading utilities for inference

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.