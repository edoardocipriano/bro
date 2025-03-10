# Tumor Classification CNN

A convolutional neural network (CNN) for classifying tumor images using PyTorch.

## Project Structure

- `src/model.py`: Contains the CNN architecture with BatchNorm and Dropout layers
- `src/data_utils.py`: Utilities for loading and preprocessing the tumor dataset
- `src/train.py`: Training script with progress bars and visualization
- `src/evaluate.py`: Evaluation utilities for model assessment and visualization
- `src/main.py`: Main entry point with CLI for running training

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

### Model Training

Train a model with specified parameters:
```bash
python src/main.py
```

Optional arguments:
- `--batch_size`: Batch size for training (default: 32)
- `--num_epochs`: Number of training epochs (default: 15)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--dropout_rate`: Dropout probability (default: 0.5)
- `--num_classes`: Number of classes (default: 9)
- `--num_workers`: Number of data loading workers (default: 4)
- `--model_type`: Model architecture to use (default: 'light')

## Results Directory Structure
```
experiments/
└── tumor_classification_[model_type]_[timestamp]/
    ├── config.json            # Model configuration
    ├── training_history.json  # Training metrics
    ├── checkpoints/
    │   ├── best_model_acc_X.pth   # Best checkpoint by accuracy
    │   ├── best_model_final.pth   # Final best model
    │   └── model_epoch_N.pth      # Checkpoint for each epoch
    └── evaluation/            # Evaluation results (if test set available)
```

### Tracked Metrics
- Training and validation accuracy
- Training and validation loss
- Best model checkpoints
- Training time
- Confusion matrices
- Classification reports

### Visualization
- Training/validation accuracy curves
- Training/validation loss curves
- Confusion matrices

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